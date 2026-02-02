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
    omega = rng.standard_normal(N)
    omega = omega - float(np.mean(omega))  # 旋转坐标系：<omega>=0
    return G, adj, omega


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

        sum_term = 0.0
        is_driver = False
        for j in neighbors:
            phase_diff = theta_star[j] - theta_star[i]
            cos_val = float(np.cos(phase_diff))

            cos_eff = cos_val - epsilon
            if cos_eff < 0.0:
                is_driver = True
                sum_term += abs(cos_eff) - cos_eff  # = 2*(epsilon - cos_val)

        if is_driver:
            drivers.append(i)
            F[i] = 1000 *K * sum_term

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

    dθ_i/dt = ω_i + K Σ_j A_ij sin(θ_j-θ_i) + F_i sin(θ_i* - θ_i)

    耦合项向量化：
    Σ_j A_ij sin(θ_j-θ_i) = cos(θ_i) (A sinθ)_i - sin(θ_i) (A cosθ)_i
    """

    s = np.sin(theta)
    c = np.cos(theta)
    a_s = adj @ s
    a_c = adj @ c
    coupling = c * a_s - s * a_c
    control = F * np.sin(theta_star - theta)
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


def plot_results(
    G: nx.Graph,
    drivers: list[int],
    unc: SimulationResult,
    con: SimulationResult,
    topology_name: str,
    layout_seed: int,
    show_relative_phase: bool = True,
) -> None:
    """画图：网络+序参量+相位轨迹（角度）。"""

    set_matplotlib_chinese()

    fig = plt.figure(figsize=(16, 10))

    ax1 = fig.add_subplot(2, 2, 1)
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

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("同步序参量 r(t) 随时间变化")
    ax2.plot(unc.t, unc.r, "k--", label="无控制", alpha=0.7)
    ax2.plot(con.t, con.r, "g-", label="有控制", linewidth=2)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xlabel("时间 t")
    ax2.set_ylabel("序参量 r")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("无控制：相位 θ_i(t)（旋转坐标系, [-π, π]）")
    theta_unc = unc.theta
    if show_relative_phase:
        theta_unc = theta_unc - unc.psi[:, None]
    theta_unc = wrap_to_pi(theta_unc)
    ax3.plot(unc.t, theta_unc, alpha=0.25, linewidth=0.8)
    ax3.set_ylim(-np.pi, np.pi)
    ax3.set_xlabel("时间 t")
    ax3.set_ylabel("相位 θ")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title("有控制：相位 θ_i(t)（减去同步相位后应收敛到 0）")
    theta_con = con.theta
    if show_relative_phase:
        theta_con = theta_con - con.psi[:, None]
    theta_con = wrap_to_pi(theta_con)
    ax4.plot(con.t, theta_con, alpha=0.25, linewidth=0.8)
    ax4.set_ylim(-np.pi, np.pi)
    ax4.set_xlabel("时间 t")
    ax4.set_ylabel("相位 θ")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main() -> None:
    # ===== 可调参数 =====
    seed = 12345
    topology: Topology = "SF"  # "ER" 或 "SF"
    N = 30
    k_mean = 10
    K = 0.2
    epsilon = 0.2  # 论文中建议的缓冲
    t_max = 50.0
    steps = 1200

    rng = np.random.default_rng(seed)

    # 1) 生成网络 + 固有频率（均值 0）
    G, adj, omega = generate_network(N=N, topology=topology, k_mean=k_mean, rng=rng)

    # 2) 计算控制策略
    theta_star, F, drivers = calculate_control(adj=adj, omega=omega, K=K, epsilon=epsilon)

    print(f"\n--- 分析结果 ({topology}) ---")
    print(f"驱动节点数量: {len(drivers)} / {N}  (比例 {len(drivers)/N:.1%})")
    print(f"最大控制增益 max(F_i): {float(np.max(F)):.6f}")

    # 3) 统一初始相位：无控制/有控制共用同一 theta0
    theta0 = rng.uniform(-np.pi, np.pi, size=N)

    print(f"\n正在模拟动力学 (K={K}, 无控制)...")
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

    # 4) 画图：相位默认绘制为 “θ_i(t) - ψ(t)” 并包裹到 [-pi, pi]
    plot_results(
        G=G,
        drivers=drivers,
        unc=unc,
        con=con,
        topology_name=topology,
        layout_seed=seed,
        show_relative_phase=True,
    )


if __name__ == "__main__":
    main()

