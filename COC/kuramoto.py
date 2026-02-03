"""Kuramoto 耦合振子网络控制示例（基于 Skardal & Arenas 方法）：

1) 绘图直接画角度（相位），并包裹到 [-pi, pi]，不再画 sin(theta)。
2) 绘图使用旋转坐标系：每个时刻减去同步相位（全局序参量的相位 psi(t)）。
3) 统一随机种子：网络、omega、初始相位 theta0 都用同一个 RNG；无控制/有控制共用同一组初值。
4) Matplotlib 中文字体支持，避免乱码。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


Topology = Literal["ER", "SF"]


def set_matplotlib_chinese() -> None:
    """尽量在 Windows 上启用中文显示，并修复负号显示。"""

    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "SimSun",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """把角度包裹到 [-pi, pi]。"""

    return (angle + np.pi) % (2 * np.pi) - np.pi


def order_parameter(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 Kuramoto 序参量 r(t) 与其相位 psi(t)。

    theta: shape (T, N)
    """

    z = np.mean(np.exp(1j * theta), axis=1)
    r = np.abs(z)
    psi = np.angle(z)
    return r, psi


def generate_network(
    N: int,
    topology: Topology,
    k_mean: int,
    rng: np.random.Generator,
    max_tries: int = 2000,
) -> tuple[nx.Graph, np.ndarray, np.ndarray]:
    """生成连通网络（ER 或 SF/BA）并初始化自然频率 omega（均值为 0）。"""

    print(f"正在生成 {topology} 网络 (N={N}, <k>≈{k_mean})...")

    if topology == "ER":
        p = k_mean / (N - 1)
        for _ in range(max_tries):
            seed = int(rng.integers(0, 2**32 - 1))
            G = nx.erdos_renyi_graph(N, p, seed=seed)
            if nx.is_connected(G):
                break
        else:
            raise RuntimeError("ER 网络在限定次数内未生成连通图，请增大 k_mean 或 max_tries")
    elif topology == "SF":
        m = max(1, int(round(k_mean / 2)))
        for _ in range(max_tries):
            seed = int(rng.integers(0, 2**32 - 1))
            G = nx.barabasi_albert_graph(N, m, seed=seed)
            if nx.is_connected(G):
                break
        else:
            raise RuntimeError("SF(BA) 网络在限定次数内未生成连通图，请调整 k_mean 或 max_tries")
    else:
        raise ValueError(f"不支持的拓扑结构: {topology}")

    adj = nx.to_numpy_array(G).astype(np.float64, copy=False)
    # 论文：频率从均值为零、方差为一的均匀分布中抽取
    # 均匀分布 U(a, b) 的方差 = (b-a)^2 / 12 = 1 => b-a = sqrt(12)
    # 均值为 0 => a = -sqrt(3), b = sqrt(3)
    omega = rng.uniform(-np.sqrt(3), np.sqrt(3), size=N)
    omega = omega - float(np.mean(omega))  # 确保均值严格为 0
    return G, adj, omega

def compute_critical_coupling(
    adj: np.ndarray,
    omega: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    """计算临界耦合强度 K_critical (Dörfler & Bullo, 2011)。

    K_critical = ||L^† ω||_{E,∞} = max_{(i,j)∈E} |[L^† ω]_i - [L^† ω]_j|

    返回:
        K_critical: 临界耦合强度
        L_dagger_omega: L^† @ omega 向量
        r_theoretical: 理论稳态序参量 (当 K >= K_critical 时)
    """
    degree = np.sum(adj, axis=1)
    L = np.diag(degree) - adj
    L_dagger = np.linalg.pinv(L)
    L_dagger_omega = L_dagger @ omega

    # 计算边无穷范数: max over all edges
    N = adj.shape[0]
    max_diff = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j] > 0:  # 存在边
                diff = abs(L_dagger_omega[i] - L_dagger_omega[j])
                if diff > max_diff:
                    max_diff = diff

    K_critical = max_diff

    # 计算理论稳态序参量: 当 K >= K_c 时，θ* = (1/K_c) L^† ω
    # 取 K = K_c 时的 theta_star 计算 r
    if K_critical > 0:
        theta_star_at_Kc = L_dagger_omega / K_critical
        z = np.mean(np.exp(1j * theta_star_at_Kc))
        r_theoretical = float(np.abs(z))
    else:
        r_theoretical = 1.0

    return K_critical, L_dagger_omega, r_theoretical

def calculate_control(
    adj: np.ndarray,
    omega: np.ndarray,
    K: float,
    epsilon: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """根据论文方法计算目标相位 theta* 和控制增益 F。

    theta* = (1/K) L^{\\dagger} omega

    稳定性判据用 Gershgorin 圆盘思想；这里用 epsilon 作缓冲：
    视 cos(Δθ) < epsilon 为“有风险”，并按 (cos-eps) 的负部来计算所需增益。
    当 epsilon=0 时，退化为论文推导里的 cos(Δθ) < 0 情况。
    """

    if K <= 0:
        raise ValueError("K 必须为正")

    N = omega.shape[0]
    degree = np.sum(adj, axis=1)

    L = np.diag(degree) - adj
    L_dagger = np.linalg.pinv(L)
    theta_star = (1.0 / K) * (L_dagger @ omega)

    F = np.zeros(N, dtype=float)
    drivers: list[int] = []

    for i in range(N):
        neighbors = np.flatnonzero(adj[i] > 0)
        if neighbors.size == 0:
            continue

        # 论文公式: F_i >= K * sum_j A_ij * [|cos(Δθ)| - cos(Δθ)]
        # 当 cos(Δθ) >= 0 时，贡献为 0
        # 当 cos(Δθ) < 0 时，贡献为 -2*cos(Δθ)
        sum_term = 0.0
        is_driver = False
        for j in neighbors:
            phase_diff = theta_star[j] - theta_star[i]
            cos_val = float(np.cos(phase_diff))

            # 使用 epsilon 缓冲
            if cos_val < epsilon:
                is_driver = True
                # |cos| - cos 当 cos < 0 时 = -2*cos
                # 当 0 < cos < epsilon 时，我们也加入一些控制
                sum_term += max(0, epsilon - cos_val) * 2

        if is_driver:
            drivers.append(i)
            # 增大安全系数到 5.0，确保快速收敛
            F[i] = 5.0 * K * sum_term

    # 额外：对所有节点施加一个"归心"控制
    # 低度数节点需要更强的控制（因为耦合弱）
    for i in range(N):
        # 基础控制力与度数成正比
        base_control = 0.5 * K * degree[i]
        # 低度数节点（叶子节点）需要额外的控制力
        if degree[i] <= 2:
            base_control = 3.0 * K  # 固定较大值
        if F[i] < base_control:
            F[i] = base_control

    return theta_star, F, drivers


def kuramoto_rhs(
    t: float,
    theta: np.ndarray,
    omega: np.ndarray,
    K: float,
    adj: np.ndarray,
    F: np.ndarray,
    theta_star: np.ndarray,
) -> np.ndarray:
    """Kuramoto 动力学（含控制项）。

    dθ_i/dt = ω_i + K Σ_j A_ij sin(θ_j-θ_i) + F_i * control_term
    
    控制项选择：
    - 论文使用 sin(θ_i* - θ_i)，但这是 2π 周期的
    - 为了避免节点"绕错方向"，我们使用 wrap_to_pi(θ_i* - θ_i)
      这样控制力总是指向最近的目标
    """

    s = np.sin(theta)
    c = np.cos(theta)
    a_s = adj @ s
    a_c = adj @ c
    coupling = c * a_s - s * a_c
    
    # 使用 wrap 后的差值，这样总是往"最近"的目标走
    diff = theta_star - theta
    diff_wrapped = (diff + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
    control = F * np.sin(diff_wrapped)
    
    return omega + K * coupling + control


@dataclass(frozen=True)
class SimulationResult:
    t: np.ndarray
    theta: np.ndarray  # shape (T, N)
    r: np.ndarray
    psi: np.ndarray


def run_simulation(
    omega: np.ndarray,
    K: float,
    adj: np.ndarray,
    F: np.ndarray,
    theta_star: np.ndarray,
    theta0: np.ndarray,
    t_max: float = 50.0,
    steps: int = 1000,
    method: str = "RK45",
) -> SimulationResult:
    """数值积分并计算序参量。"""

    t_eval = np.linspace(0.0, float(t_max), int(steps))

    sol = solve_ivp(
        fun=lambda t, y: kuramoto_rhs(t, y, omega, K, adj, F, theta_star),
        t_span=(0.0, float(t_max)),
        y0=theta0,
        t_eval=t_eval,
        method=method,
        rtol=1e-6,
        atol=1e-8,
    )
    if not sol.success:
        raise RuntimeError(f"ODE 求解失败: {sol.message}")

    theta_tn = sol.y.T  # (T, N)
    r, psi = order_parameter(theta_tn)
    return SimulationResult(t=sol.t, theta=theta_tn, r=r, psi=psi)


def compute_jacobian(
    adj: np.ndarray,
    theta_star: np.ndarray,
    K: float,
    F: np.ndarray | None = None,
) -> np.ndarray:
    """计算同步态处的 Jacobian（对 Kuramoto + 可选控制项）。"""

    N = adj.shape[0]
    df = np.zeros((N, N), dtype=float)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if adj[i, j] == 0:
                continue
            cos_val = float(np.cos(theta_star[j] - theta_star[i]))
            df[i, j] = K * cos_val

    for i in range(N):
        df[i, i] = -float(np.sum(df[i, :]))

    if F is not None:
        df = df.copy()
        df[np.diag_indices(N)] -= F

    return df


def plot_results(
    G: nx.Graph,
    drivers: list[int],
    unc: SimulationResult,
    con: SimulationResult,
    topology_name: str,
    layout_seed: int,
    theta_star: np.ndarray,
    adj: np.ndarray,
    K: float,
    F: np.ndarray,
    show_relative_phase: bool = True,
) -> None:
    """画图：网络+序参量+相位轨迹（角度）+ Jacobian。"""

    set_matplotlib_chinese()

    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_title(f"网络拓扑: {topology_name}（红色为驱动节点）")
    node_colors = ["red" if i in set(drivers) else "skyblue" for i in range(G.number_of_nodes())]
    pos = nx.spring_layout(G, seed=layout_seed)
    nx.draw(
        G,
        pos,
        ax=ax1,
        node_color=node_colors,
        node_size=60,
        edge_color="gray",
        alpha=0.7,
        linewidths=0.0,
    )

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_title("同步序参量 r(t) 随时间变化")
    ax2.plot(unc.t, unc.r, "k--", label="无控制", alpha=0.7)
    ax2.plot(con.t, con.r, "g-", label="有控制", linewidth=2)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xlabel("时间 t")
    ax2.set_ylabel("序参量 r")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = fig.add_subplot(2, 3, 4)
    ax3.set_title("无控制：相位 (θ_i - ψ) - (θ_i* - ψ*)")
    theta_unc = unc.theta
    if show_relative_phase:
        # 消除全局相位漂移：用圆平均对齐
        # psi(t) = arg(mean(exp(i*theta)))
        # psi* = arg(mean(exp(i*theta*)))
        psi_star = float(np.angle(np.mean(np.exp(1j * theta_star))))
        # 每个时刻减去 (psi(t) - psi*)
        global_drift = unc.psi - psi_star  # shape (T,)
        theta_unc = theta_unc - theta_star[None, :] - global_drift[:, None]
    theta_unc = wrap_to_pi(theta_unc)
    ax3.plot(unc.t, theta_unc, alpha=0.25, linewidth=0.8)
    ax3.set_ylim(-np.pi, np.pi)
    ax3.set_xlabel("时间 t")
    ax3.set_ylabel("相位 θ")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 3, 5)
    ax4.set_title("有控制：相位 (θ_i - ψ) - (θ_i* - ψ*)（应收敛到 0）")
    theta_con = con.theta
    if show_relative_phase:
        # 消除全局相位漂移
        psi_star = float(np.angle(np.mean(np.exp(1j * theta_star))))
        global_drift = con.psi - psi_star
        theta_con = theta_con - theta_star[None, :] - global_drift[:, None]
    theta_con = wrap_to_pi(theta_con)
    ax4.plot(con.t, theta_con, alpha=0.25, linewidth=0.8)
    ax4.set_ylim(-np.pi, np.pi)
    ax4.set_xlabel("时间 t")
    ax4.set_ylabel("相位 θ")
    ax4.grid(True, alpha=0.3)

    # Jacobian 热力图（无控制）
    ax5 = fig.add_subplot(2, 3, 3)
    ax5.set_title("Jacobian DF (无控制)")
    df_no_ctrl = compute_jacobian(adj=adj, theta_star=theta_star, K=K, F=None)
    vmax = float(np.max(np.abs(df_no_ctrl))) if df_no_ctrl.size else 1.0
    im1 = ax5.imshow(df_no_ctrl, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    fig.colorbar(im1, ax=ax5, fraction=0.046, pad=0.04)
    # 添加网格线使每个单元格更清晰
    N = df_no_ctrl.shape[0]
    ax5.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax5.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax5.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    ax5.tick_params(which="minor", size=0)
    ax5.set_xticks(np.arange(0, N, max(1, N // 5)))
    ax5.set_yticks(np.arange(0, N, max(1, N // 5)))
    ax5.set_xlabel("j")
    ax5.set_ylabel("i")

    # Jacobian 热力图（有控制）
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_title("Jacobian DF (有控制)")
    df_ctrl = compute_jacobian(adj=adj, theta_star=theta_star, K=K, F=F)
    vmax2 = float(np.max(np.abs(df_ctrl))) if df_ctrl.size else 1.0
    im2 = ax6.imshow(df_ctrl, cmap="RdBu_r", vmin=-vmax2, vmax=vmax2, aspect="equal")
    fig.colorbar(im2, ax=ax6, fraction=0.046, pad=0.04)
    ax6.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax6.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax6.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    ax6.tick_params(which="minor", size=0)
    ax6.set_xticks(np.arange(0, N, max(1, N // 5)))
    ax6.set_yticks(np.arange(0, N, max(1, N // 5)))
    ax6.set_xlabel("j")
    ax6.set_ylabel("i")

    plt.tight_layout()
    plt.show()


def main() -> None:
    # ===== 论文参数 (Skardal & Arenas) =====
    # 论文使用 N=1000, <k>=6, K=0.4
    # 但关键是 K 必须 >= K_critical 才能有稳定的同步态！
    seed = 12345
    topology: Topology = "ER"  # "ER" 或 "SF"
    N = 1000
    k_mean = 4  # 平均度数
    epsilon = 0.2  # 论文中建议的缓冲
    t_max = 100.0
    steps = 3000
    
    # K 将在计算 K_critical 后自动设置

    rng = np.random.default_rng(seed)

    # 1) 生成网络 + 固有频率（均值 0）
    G, adj, omega = generate_network(N=N, topology=topology, k_mean=k_mean, rng=rng)

    # 2) 计算临界耦合强度（Dörfler & Bullo 理论）
    K_critical, L_dagger_omega, r_theoretical = compute_critical_coupling(adj=adj, omega=omega)

    # 计算使 theta* 跨度 < 2π 的最小 K
    theta_span = L_dagger_omega.max() - L_dagger_omega.min()
    K_min_for_2pi = theta_span / (2 * np.pi)
    
    # 要有驱动节点，需要某些边的 cos(Δθ*) < epsilon=0.2
    # 即 |Δθ*| > arccos(0.2) ≈ 1.37 rad
    # 即 K < K_critical / 1.37 ≈ 0.73 * K_critical
    # 同时要保证 theta* 跨度 < 2π，即 K > K_min_for_2pi
    # K = max(K_min_for_2pi * 1.1, 0.65 * K_critical)
    K=1

    print(f"\n{'='*60}")
    print(f"  临界耦合分析")
    print(f"{'='*60}")
    print(f"  K_critical (边上最大相位差=1rad) = {K_critical:.4f}")
    print(f"  K_min (使θ*跨度<2π) = {K_min_for_2pi:.4f}")
    print(f"  选择 K = {K:.4f}")
    print(f"  K / K_critical = {K / K_critical:.2f}")
    theta_star_test = L_dagger_omega / K
    span_deg = np.degrees(theta_star_test.max() - theta_star_test.min())
    print(f"  θ* 跨度 = {span_deg:.0f}° (需 < 360°)")
    max_phase_diff = K_critical / K
    print(f"  边上最大相位差 = {max_phase_diff:.2f} rad ({np.degrees(max_phase_diff):.0f}°)")
    print(f"{'='*60}")

    # 3) 计算控制策略
    theta_star, F, drivers = calculate_control(adj=adj, omega=omega, K=K, epsilon=epsilon)

    print(f"\n--- Skardal & Arenas 控制分析 ({topology}) ---")
    print(f"驱动节点数量: {len(drivers)} / {N}  (比例 {len(drivers)/N:.1%})")
    print(f"最大控制增益 max(F_i): {float(np.max(F)):.6f}")
    # 计算稳态序参量
    z_star = np.mean(np.exp(1j * theta_star))
    r_star = float(np.abs(z_star))
    print(f"目标稳态序参量 r(θ*) = {r_star:.4f}")

    # 4) 统一初始相位：无控制/有控制共用同一 theta0
    theta0 = rng.uniform(-np.pi, np.pi, size=N)

    print(f"\n正在模拟动力学 (K={K}, 无控制)...")
    print(f"  注：无控制时若 K < K_critical，系统无法达到稳定同步态")
    unc = run_simulation(
        omega=omega,
        K=K,
        adj=adj,
        F=np.zeros(N),
        theta_star=theta_star,
        theta0=theta0,
        t_max=t_max,
        steps=steps,
    )

    print(f"正在模拟动力学 (K={K}, 有控制)...")
    con = run_simulation(
        omega=omega,
        K=K,
        adj=adj,
        F=F,
        theta_star=theta_star,
        theta0=theta0,
        t_max=t_max,
        steps=steps,
    )

    # 5) 画图：相位绘制为 "θ_i(t) - θ_i*" 并包裹到 [-pi, pi]
    plot_results(
        G=G,
        drivers=drivers,
        unc=unc,
        con=con,
        topology_name=topology,
        layout_seed=seed,
        theta_star=theta_star,
        adj=adj,
        K=K,
        F=F,
        show_relative_phase=True,
    )


if __name__ == "__main__":
    main()

