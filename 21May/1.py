import numpy as np
import matplotlib.pyplot as plt

# === Simulation Parameters ===
N = 100000  # Number of users
alpha = 3.5  # Path loss exponent
qos_threshold = 0.0001  # Threshold for QoS failure
P_total = 1.0  # Total power
sigma_squared = 1e-9  # Noise power

# === Generate realistic h values (Rayleigh + Path Loss) ===
np.random.seed(42)
d = np.random.uniform(50, 500, N)
g = np.random.rayleigh(scale=1.0, size=N)
h = g / (d ** (alpha / 2))

# === Sort h and users ===
sorted_indices = np.argsort(h)
sorted_h = h[sorted_indices]
sorted_users = np.arange(1, N + 1)[sorted_indices]

# === Clustering Methods ===
def form_clusters(sorted_h, method='optimal'):
    clusters = []
    if method == 'optimal':
        for i in range(N // 2):
            h1 = sorted_h[i]
            h2 = sorted_h[-(i + 1)]
            clusters.append((min(h1, h2), max(h1, h2)))
    elif method == 'balanced':
        mid = N // 2
        for i in range(mid):
            h1 = sorted_h[i]
            h2 = sorted_h[i + mid]
            clusters.append((min(h1, h2), max(h1, h2)))
    return clusters

# === NOMA Rate Calculation ===
def noma_rate(h1, h2, P_total):
    P1 = 0.8 * P_total
    P2 = 0.2 * P_total
    R1 = np.log2(1 + (P1 * h1) / (P2 * h1 + sigma_squared))
    R2 = np.log2(1 + (P2 * h2) / sigma_squared)
    return R1, R2

# === Analyze Clusters ===
def analyze_clusters(clusters):
    h_diffs = []
    qos_failures = 0
    total_rate = 0
    for h1, h2 in clusters:
        diff = abs(h2 - h1)
        h_diffs.append(diff)
        if diff < qos_threshold:
            qos_failures += 1
        R1, R2 = noma_rate(h1, h2, P_total)
        total_rate += R1 + R2
    avg_rate = total_rate / len(clusters)
    return h_diffs, qos_failures, avg_rate

# === Run Analysis ===
optimal_clusters = form_clusters(sorted_h, method='optimal')
balanced_clusters = form_clusters(sorted_h, method='balanced')

opt_h_diffs, opt_qos_failures, opt_avg_rate = analyze_clusters(optimal_clusters)
bal_h_diffs, bal_qos_failures, bal_avg_rate = analyze_clusters(balanced_clusters)

# === Plotting ===
cluster_indices = np.arange(1, len(opt_h_diffs) + 1)
plt.figure(figsize=(12, 6))
plt.plot(cluster_indices, opt_h_diffs, 'o-', label='Optimal Clustering', color='green')
plt.plot(cluster_indices, bal_h_diffs, 's-', label='Balanced Clustering', color='blue')
plt.axhline(y=qos_threshold, color='red', linestyle='--', label='QoS Threshold (0.0001)')
plt.title("Channel Gain Differences per Cluster in PD-NOMA")
plt.xlabel("Cluster Index")
plt.ylabel("Δh = |h₂ - h₁|")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Print Summary ===
print(f"Optimal Clustering: Avg Rate = {opt_avg_rate:.4f}, QoS Failures = {opt_qos_failures}")
print(f"Balanced Clustering: Avg Rate = {bal_avg_rate:.4f}, QoS Failures = {bal_qos_failures}")
