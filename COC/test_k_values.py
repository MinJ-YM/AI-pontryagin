import numpy as np
import networkx as nx

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
epsilon = 0.2

print("=" * 70)
print("K值对比分析：揭示反直觉现象的原因")
print("=" * 70)

for K_ratio in [0.2, 0.3, 0.4]:
    K = K_ratio * Kc
    theta_star = (1.0 / K) * (L_dagger @ omega)
    
    # 计算振子4的增益
    neighbors_4 = np.flatnonzero(adj[4] > 0)
    sum_gain = 0
    problem_edges = []
    
    for j in neighbors_4:
        phase_diff = theta_star[j] - theta_star[4]
        cos_val = np.cos(phase_diff)
        cos_eff = cos_val - epsilon
        if cos_eff < 0:
            sum_gain += abs(cos_eff) - cos_eff
            problem_edges.append((j, phase_diff, cos_val))
    
    F_4 = 1.2 * K * sum_gain
    is_driver = F_4 > 1e-9
    span = theta_star.max() - theta_star.min()
    
    print(f"\nK = {K_ratio:.1f} × Kc = {K:.4f}:")
    print(f"  theta_star 跨度: {span:.2f} rad ({np.degrees(span):.0f}°)")
    print(f"  theta*_4 = {theta_star[4]:.3f} rad")
    print(f"  问题边数: {len(problem_edges)}/{len(neighbors_4)}")
    if problem_edges:
        print(f"  问题边详情:")
        for j, diff, cos_v in problem_edges:
            print(f"    -> 邻居{j}: Δθ*={diff:.3f} rad, cos={cos_v:.3f}")
    print(f"  sum_gain = {sum_gain:.6f}")
    print(f"  F_4 = {F_4:.6f}")
    print(f"  是否为driver: {'是' if is_driver else '否 ⚠️'}")

print("\n" + "=" * 70)
print("结论:")
print("  K越大 → theta_star跨度越小 → 相邻节点相位差减小")
print("  → cos(相位差)增大 → cos - epsilon 可能从负变正")
print("  → 原本的driver节点失去控制增益 → 反而不稳定！")
print("=" * 70)
