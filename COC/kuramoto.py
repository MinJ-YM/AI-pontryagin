from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

SEED = 42
Topology = Literal["ER", "SF"]
# ==========================================
# 1. 基础工具与网络生成
# ==========================================

def set_matplotlib_chinese() -> None:
    """配置 Matplotlib 中文显示。"""
    plt.rcParams["font.sans-serif"] = [
        "SimHei", "Microsoft YaHei", "PingFang SC", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """将角度包裹到 [-pi, pi]。"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def order_parameter(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """计算序参量 r(t) 和平均相位 psi(t)。"""
    z = np.mean(np.exp(1j * theta), axis=1)
    return np.abs(z), np.angle(z)

def generate_network(
    N: int,
    topology: Topology,
    k_mean: int,
    rng: np.random.Generator,
    max_tries: int = 2000,
) -> tuple[nx.Graph, np.ndarray, np.ndarray]:
    """生成连通网络并初始化自然频率（强制去均值）。"""
    print(f"正在生成 {topology} 网络 (N={N}, <k>≈{k_mean})...")

    if topology == "ER":
        p = k_mean / (N - 1)
        for _ in range(max_tries):
            G = nx.erdos_renyi_graph(N, p, seed = int(rng.integers(0, 2**32 - 1)))
            if nx.is_connected(G): break
        else: raise RuntimeError("无法生成连通 ER 图")
    elif topology == "SF":
        m = max(1, int(round(k_mean / 2)))
        for _ in range(max_tries):
            G = nx.barabasi_albert_graph(N, m, seed = int(rng.integers(0, 2**32 - 1)))
            if nx.is_connected(G): break
        else: raise RuntimeError("无法生成连通 SF 图")
    else:
        raise ValueError(f"未知拓扑: {topology}")

    adj = nx.to_numpy_array(G).astype(np.float64)
    
    # 生成频率并去均值
    omega = rng.uniform(-1, 1, size=N) # 均匀分布
    ## 关键：确保 sum(omega) = 0
    
    return G, adj, omega

# ==========================================
# 2. 核心数学计算 (伪逆与临界耦合)
# ==========================================

def compute_critical_coupling(
    adj: np.ndarray,
    omega: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    计算临界耦合强度 Kc 和 归一化的 theta_star_base。
    
    K_critical = || L† ω ||_{edge, ∞}
    即当 K = Kc 时，网络中最紧张的那条边的相位差刚好是 1 rad。
    """
    # 1. 确保 omega 是相对值（零均值）
    # omega_centered = omega - np.mean(omega)
    
    # 2. 构建拉普拉斯矩阵
    degree = np.sum(adj, axis=1)
    L = np.diag(degree) - adj
    
    # 3. 计算 L 的伪逆乘以 omega
    # 数学性质：如果 sum(omega)=0，那么 sum(L_dagger_omega)=0
    L_dagger = np.linalg.pinv(L)
    L_dagger_omega = L_dagger @ omega
    
    # 4. 遍历所有边，找到相位差最大值
    # theta_i - theta_j = (L†ω)_i/K - (L†ω)_j/K
    # 临界条件是 max |theta_i - theta_j| = 1
    rows, cols = np.nonzero(np.triu(adj)) # 只取上三角，避免重复
    max_diff = 0.0
    if len(rows) > 0:
        diffs = np.abs(L_dagger_omega[rows] - L_dagger_omega[cols])
        max_diff = np.max(diffs)
    
    K_critical = max_diff if max_diff > 1e-9 else 1.0
    
    return K_critical, L_dagger_omega

def calculate_control(
    adj: np.ndarray,
    omega: np.ndarray,
    K: float,
    epsilon: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    N = len(omega)
    
    # 1. 再次确保去均值，保证计算出的 theta_star 重心在 0
    # omega_centered = omega - np.mean(omega)
    
    # 2. 计算 theta* = (1/K) * L† * ω
    degree = np.sum(adj, axis=1)
    L = np.diag(degree) - adj
    L_dagger = np.linalg.pinv(L)
    
    # 这里得到的 theta_star 必然满足 sum(theta_star) ≈ 0
    # 最容易导致控制出现问题的地方：这里就要求theta之间足够小要可以线性化
    theta_star = (1.0 / K) * (L_dagger @ omega)
    
    # 3. 识别驱动节点 + 计算增益 (Gershgorin + ε 缓冲)
    F = np.zeros(N)
    drivers = []
    
    for i in range(N):
        neighbors = np.flatnonzero(adj[i] > 0)
        sum_gain = 0.0
        is_driver = False
        
        for j in neighbors:
            phase_diff = theta_star[j] - theta_star[i]
            cos_val = float(np.cos(phase_diff))

            # ε 缓冲：保守估计（把 cos 往负方向平移 ε）
            cos_eff = cos_val - epsilon
            if cos_eff < 0:
                is_driver = True
                # Gershgorin 充分条件对应的增益项：|cos_eff| - cos_eff
                sum_gain += (abs(cos_eff) - cos_eff)
        
        if is_driver and sum_gain > 0:
            drivers.append(i)
            F[i] =  K * sum_gain
            
    return theta_star, F, drivers

# ==========================================
# 3. 动力学仿真
# ==========================================

def kuramoto_rhs(
    t: float,
    theta: np.ndarray,
    omega: np.ndarray,
    K: float,
    adj: np.ndarray,
    F: np.ndarray,
    theta_star: np.ndarray
) -> np.ndarray:
    # 耦合项
    # 利用 sin(j - i) = -sin(i - j)
    # delta_theta[i, j] = theta[i] - theta[j]
    delta = theta[:, None] - theta[None, :]
    coupling = np.sum(adj * np.sin(-delta), axis=1)
    _,theta_mid= order_parameter(theta[None,:])
    
    # 控制项 
    diff_wrapped = wrap_to_pi(theta_mid - theta)
    control = 1.5 * F * np.sin(diff_wrapped)
    
    return omega + K * coupling + control

@dataclass
class SimResult:
    t: np.ndarray
    theta: np.ndarray
    r: np.ndarray
    psi: np.ndarray

def run_simulation(
    omega: np.ndarray, K: float, adj: np.ndarray,
    F: np.ndarray, theta_star: np.ndarray, theta0: np.ndarray,
    t_max: float = 50.0
) -> SimResult:
    
    sol = solve_ivp(
        fun=lambda t, y: kuramoto_rhs(t, y, omega, K, adj, F, theta_star),
        t_span=(0, t_max),
        y0=theta0, # 确保初始条件也是去均值的
        t_eval=np.linspace(0, t_max, 2000),
        method='RK45'
    )
    
    theta_t = sol.y.T # (Time, N)
    
    r, psi = order_parameter(theta_t)
    # print(f'仿真完成: 平均相位（算数平均）={np.mean(theta_t, axis=1)}')
    # print(f'仿真完成: 平均相位={psi}')
    return SimResult(sol.t, theta_t, r, psi)

# ==========================================
# 4. 绘图与主程序
# ==========================================

def main():
    set_matplotlib_chinese()
    
    # --- 参数 ---
    N = 500
    k_mean = 6
    topology = "ER"
    
    rng = np.random.default_rng(SEED)
    
    # 1. 生成网络
    G, adj, omega = generate_network(N, topology, k_mean, rng)
    omega_centered =np.mean(omega)
    omega = omega - omega_centered
    # 2. 确定 K 值
    # 先算临界耦合 Kc
    Kc, L_dagger_omega = compute_critical_coupling(adj, omega)
    
    # 设定 K < Kc 以使得自然状态不稳定
    # 比如取 0.6 倍 Kc，这样会有很多相位差 > 1 rad 的边
    K = 0.38 * Kc
    
    # 3. 计算控制
    theta_star, F, drivers = calculate_control(adj, omega, K, epsilon=0.5)#理论上夹角不超过60度
    
    # 验证 theta_star 重心是否为 0
    center_mass = np.mean(theta_star)
    print(f"\n参数分析:")
    print(f"  K_critical = {Kc:.4f}")
    print(f"  当前 K     = {K:.4f}")
    print(f"  θ* 重心    = {center_mass:.4e} (应接近 0)")
    print(f"  驱动节点数 = {len(drivers)}")
    nonzero_F = int(np.count_nonzero(F > 1e-12))
    print(f"  非零增益数 = {nonzero_F} (min={float(np.min(F)):.3e}, max={float(np.max(F)):.3e})")
    
    # 4. 仿真
    theta0 = rng.uniform(-np.pi, np.pi, N)
    
    # 无控制 (F=0)
    print("运行无控制仿真...")
    res_unc = run_simulation(omega, K, adj, np.zeros(N), theta_star, theta0)
    
    # 有控制
    print("运行有控制仿真...")
    res_con = run_simulation(omega, K, adj, F, theta_star, theta0)

    # 诊断：检查末段是否仍有明显震荡的振子
    tail_frac = 0.2
    tail_start = int((1.0 - tail_frac) * len(res_con.t))
    tail_diff = wrap_to_pi(res_con.theta[tail_start:, :] - theta_star[None, :])
    tail_std = np.std(tail_diff, axis=0)
    tail_max = np.max(np.abs(tail_diff), axis=0)
    unstable_idx = np.where(tail_std > 0.2)[0]

    print("\n末段相位收敛诊断 (相对 θ*):")
    print(f"  震荡阈值: std > 0.2 rad")
    print(f"  仍明显震荡的振子数: {len(unstable_idx)}")
    if len(unstable_idx) > 0:
        preview = ", ".join(map(str, unstable_idx[:20]))
        more = "..." if len(unstable_idx) > 20 else ""
        print(f"  震荡振子索引: {preview}{more}")
        worst = int(unstable_idx[np.argmax(tail_std[unstable_idx])])
        print(f"  最大 std 振子: {worst}, std={tail_std[worst]:.3f}, max|diff|={tail_max[worst]:.3f}")
    
    # 5. 绘图（使用彩色轨迹）
    fig = plt.figure(figsize=(18, 10))
    
    # 子图1: 网络
    ax1 = fig.add_subplot(2, 2, 1)
    pos = nx.spring_layout(G, seed=SEED)
    colors = ['red' if i in drivers else 'skyblue' for i in range(N)]
    nx.draw(G, pos, ax=ax1, node_size=50, node_color=colors, edge_color='#ddd')
    ax1.set_title(f"网络拓扑 (红色=驱动节点, N={N})")
    
    # 子图2: 序参量
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(res_unc.t, res_unc.r, 'k--', alpha=0.6, label='无控制')
    ax2.plot(res_con.t, res_con.r, 'r-', linewidth=2, label='有控制')
    ax2.set_ylim(0, 1.05)
    ax2.set_title(f"序参量 r(t) (K={K:.2f})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 无控制的相位 (相对相位) - 彩色轨迹
    # 因为无控制时中心会漂移，为了看清是否散开，我们要减去重心 psi
    ax3 = fig.add_subplot(2, 2, 3)
    phases_unc = wrap_to_pi(res_unc.theta - res_unc.psi[:, None])
    
    # 使用彩色映射绘制每条轨迹
    cmap = plt.get_cmap("rainbow")
    for i in range(N):
        color = cmap(i / N)
        ax3.plot(res_unc.t, phases_unc[:, i], color=color, alpha=0.6, linewidth=1.0)
    
    ax3.set_title("无控制: 相对相位 (θ_i - ψ)")
    ax3.set_ylim(-np.pi, np.pi)
    ax3.set_ylabel("相位 (rad)")
    ax3.set_xlabel("时间")
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 有控制的相位 (相对重心) - 彩色轨迹
    # 这里画 theta - psi，和无控制保持一致
    ax4 = fig.add_subplot(2, 2, 4)
    phases_con = wrap_to_pi(res_con.theta - res_con.psi[:, None])

    # 使用彩色映射绘制每条轨迹
    for i in range(N):
        color = cmap(i / N)
        ax4.plot(res_con.t, phases_con[:, i], color=color, alpha=0.6, linewidth=0.8)

    ax4.set_title("有控制: 相对相位 (θ_i - ψ)")
    ax4.set_ylim(-np.pi, np.pi)
    ax4.set_xlabel("时间")
    ax4.set_ylabel("相位 (rad)")
    ax4.axhline(0, color='blue', linestyle=':', alpha=0.5, linewidth=2, label="中心 0")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()