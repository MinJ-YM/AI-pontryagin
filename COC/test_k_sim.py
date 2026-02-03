import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

SEED = 42
rng = np.random.default_rng(SEED)
p = 6 / 99

for _ in range(2000):
    G = nx.erdos_renyi_graph(100, p, seed=int(rng.integers(0, 1_000_000_000)))
    if nx.is_connected(G): 
        break

adj = nx.to_numpy_array(G).astype(np.float64)
omega = rng.uniform(-1, 1, size=100)
omega -= np.mean(omega)
degree = np.sum(adj, axis=1)
L = np.diag(degree) - adj
L_dagger = np.linalg.pinv(L)
Kc = 0.4975

def kuramoto_rhs(t, theta, omega, K, adj, F):
    delta = theta[:, None] - theta[None, :]
    coupling = np.sum(adj * np.sin(-delta), axis=1)
    diff_wrapped = wrap_to_pi(0 - theta)
    control = 2 * F * np.sin(diff_wrapped)
    return omega + K * coupling + control

print("=" * 70)
print("仿真结果对比：不同K值下的实际表现")
print("=" * 70)

theta0 = rng.uniform(-np.pi, np.pi, 100)

for K_ratio in [0.2, 0.3, 0.4]:
    K = K_ratio * Kc
    theta_star = (1.0 / K) * (L_dagger @ omega)
    
    # 计算增益
    epsilon = 0.2
    F = np.zeros(100)
    for i in range(100):
        neighbors = np.flatnonzero(adj[i] > 0)
        sum_gain = 0
        for j in neighbors:
            cos_eff = np.cos(theta_star[j] - theta_star[i]) - epsilon
            if cos_eff < 0:
                sum_gain += abs(cos_eff) - cos_eff
        if sum_gain > 0:
            F[i] = 1.2 * K * sum_gain
    
    # 仿真
    sol = solve_ivp(
        fun=lambda t, y: kuramoto_rhs(t, y, omega, K, adj, F),
        t_span=(0, 50),
        y0=theta0,
        t_eval=np.linspace(0, 50, 2000),
        method='RK45'
    )
    
    theta_final = wrap_to_pi(sol.y[:, -1])
    
    # 计算末段统计
    tail_theta = wrap_to_pi(sol.y[:, -400:].T)  # 最后20%
    tail_std = np.std(tail_theta, axis=0)
    unstable = np.where(tail_std > 0.2)[0]
    
    # 序参量
    z = np.mean(np.exp(1j * sol.y[:, -1]))
    r_final = abs(z)
    
    print(f"\nK = {K_ratio:.1f} × Kc = {K:.4f}:")
    print(f"  F_4 = {F[4]:.6f}")
    print(f"  末段 theta_4 = {theta_final[4]:.3f} rad ({np.degrees(theta_final[4]):.1f}°)")
    print(f"  末段 std(theta_4) = {tail_std[4]:.3f}")
    print(f"  不稳定振子数: {len(unstable)}")
    if len(unstable) > 0:
        print(f"    索引: {unstable.tolist()[:10]}")
    print(f"  末态序参量 r = {r_final:.4f}")

print("=" * 70)
