# optimal +Balanced + blosoom

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# --- Parameters ---
N = 100
alpha = 3.5
qos_threshold = 0.0001
P_total = 1.0
sigma_squared = 1e-9

# --- Channel gain simulation ---
np.random.seed(42)
d = np.random.uniform(50, 500, N)
g = np.random.rayleigh(scale=1.0, size=N)
h = g / (d ** (alpha / 2))
sorted_h = np.sort(h)

# --- NOMA rate function ---
def noma_rate(h1, h2, P_total):
    P1 = 0.8 * P_total
    P2 = 0.2 * P_total
    R1 = np.log2(1 + (P1 * h1) / (P2 * h1 + sigma_squared))
    R2 = np.log2(1 + (P2 * h2) / sigma_squared)
    return R1, R2

# --- Cluster analyzers ---
def analyze_clusters(pairs):
    h_diffs, qos_failures, total_rate = [], 0, 0
    for h1, h2 in pairs:
        diff = abs(h2 - h1)
        h_diffs.append(diff)
        if diff < qos_threshold:
            qos_failures += 1
        R1, R2 = noma_rate(h1, h2, P_total)
        total_rate += R1 + R2
    avg_rate = total_rate / len(pairs)
    return h_diffs, qos_failures, avg_rate

# --- Clustering strategies ---
def optimal_pairs(h_vals):
    return [(h_vals[i], h_vals[-(i+1)]) for i in range(len(h_vals)//2)]

def balanced_pairs(h_vals):
    mid = len(h_vals) // 2
    return [(h_vals[i], h_vals[i+mid]) for i in range(mid)]

def blossom_pairs(h_vals):
    G = nx.Graph()
    for i in range(len(h_vals)):
        for j in range(i+1, len(h_vals)):
            G.add_edge(i, j, weight=abs(h_vals[i] - h_vals[j]))
    matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)
    used = set()
    pairs = []
    for i, j in matching:
        if i not in used and j not in used:
            pairs.append((min(h_vals[i], h_vals[j]), max(h_vals[i], h_vals[j])))
            used.update([i, j])
    return pairs

# --- Generate cluster pairs ---
opt_pairs = optimal_pairs(sorted_h)
bal_pairs = balanced_pairs(sorted_h)
blo_pairs = blossom_pairs(sorted_h)

# --- Analyze all clusters ---
opt_diffs, opt_qos, opt_rate = analyze_clusters(opt_pairs)
bal_diffs, bal_qos, bal_rate = analyze_clusters(bal_pairs)
blo_diffs, blo_qos, blo_rate = analyze_clusters(blo_pairs)

# --- PLOT 1: Unsorted Δh ---
cluster_indices = np.arange(1, len(opt_diffs)+1)

plt.figure(figsize=(12, 6))
plt.plot(cluster_indices, opt_diffs, 'o-', label='Optimal Clustering', color='green')
plt.plot(cluster_indices, bal_diffs, 's-', label='Balanced Clustering', color='blue')
plt.plot(cluster_indices, blo_diffs, 'd-', label='Blossom Clustering', color='orange')
plt.axhline(y=qos_threshold, color='red', linestyle='--', label='QoS Threshold (0.0001)')
plt.title("Channel Gain Differences per Cluster (Unsorted)")
plt.xlabel("Cluster Index")
plt.ylabel("Δh = |h₂ - h₁|")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- PLOT 2: Sorted Δh ---
plt.figure(figsize=(12, 6))
plt.plot(cluster_indices, sorted(opt_diffs), 'o-', label='Optimal Clustering', color='green')
plt.plot(cluster_indices, sorted(bal_diffs), 's-', label='Balanced Clustering', color='blue')
plt.plot(cluster_indices, sorted(blo_diffs), 'd-', label='Blossom Clustering', color='orange')
plt.axhline(y=qos_threshold, color='red', linestyle='--', label='QoS Threshold (0.0001)')
plt.title("Sorted Channel Gain Differences per Cluster")
plt.xlabel("Sorted Cluster Index")
plt.ylabel("Δh = |h₂ - h₁| (Sorted)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Print summary ---
print("\n===== SUMMARY =====")
print(f"Optimal   → Avg Rate: {opt_rate:.4f}, QoS Failures: {opt_qos}")
print(f"Balanced  → Avg Rate: {bal_rate:.4f}, QoS Failures: {bal_qos}")
print(f"Blossom   → Avg Rate: {blo_rate:.4f}, QoS Failures: {blo_qos}")
