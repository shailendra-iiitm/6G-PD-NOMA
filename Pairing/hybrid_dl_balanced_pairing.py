import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# === Parameters ===
N = 100
alpha = 3.5
qos_threshold = 0.0001
P_total = 1.0
sigma_squared = 1e-9

# === Simulate h values ===
np.random.seed(42)
d = np.random.uniform(50, 500, N)
g = np.random.rayleigh(scale=1.0, size=N)
h = g / (d ** (alpha / 2))
h = h.reshape(-1, 1)

# === Normalize h values ===
scaler = MinMaxScaler()
h_scaled = scaler.fit_transform(h)

# === Autoencoder ===
input_dim = h_scaled.shape[1]
encoding_dim = 2
input_layer = Input(shape=(input_dim,))
encoded = Dense(8, activation='relu')(input_layer)
encoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(8, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)
autoencoder = Model(inputs=input_layer, outputs=decoded)
encoder = Model(inputs=input_layer, outputs=encoded)
autoencoder.compile(optimizer=Adam(0.01), loss='mse')
autoencoder.fit(h_scaled, h_scaled, epochs=100, batch_size=10, verbose=0)

# === Utility functions
def noma_rate(h1, h2):
    P1 = 0.8 * P_total
    P2 = 0.2 * P_total
    R1 = np.log2(1 + (P1 * h1) / (P2 * h1 + sigma_squared))
    R2 = np.log2(1 + (P2 * h2) / sigma_squared)
    return R1, R2

def analyze_clusters(pairs):
    h_diffs, qos_failures, total_rate = [], 0, 0
    for h1, h2 in pairs:
        diff = abs(h2 - h1)
        h_diffs.append(diff)
        if diff < qos_threshold:
            qos_failures += 1
        R1, R2 = noma_rate(h1, h2)
        total_rate += R1 + R2
    avg_rate = total_rate / len(pairs) if len(pairs) > 0 else 0
    return h_diffs, qos_failures, avg_rate

def match_clusters(c1, c2, used_set):
    pairs = []
    low = sorted(c1, key=lambda x: x[1])
    high = sorted(c2, key=lambda x: x[1], reverse=True)
    for a, b in zip(low, high):
        if a[0] not in used_set and b[0] not in used_set:
            pairs.append((min(a[1], b[1]), max(a[1], b[1])))
            used_set.update([a[0], b[0]])
    return pairs

# === Clustering
features = encoder.predict(h_scaled)
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(features)

clusters = [[] for _ in range(4)]
for i in range(N):
    clusters[labels[i]].append((i, h[i][0]))

clusters = sorted(clusters, key=lambda c: np.mean([x[1] for x in c]))

# DL pairing: 0<->2 and 1<->3
used = set()
dl_pairs = match_clusters(clusters[0], clusters[2], used) + match_clusters(clusters[1], clusters[3], used)

# Get unpaired users
unpaired_indices = [i for i in range(N) if i not in used]
unpaired_h = [(i, h[i][0]) for i in unpaired_indices]
unpaired_h_sorted = sorted(unpaired_h, key=lambda x: x[1])

# --- Fallback 1: Balanced ---
fallback_pairs_balanced = []
half = len(unpaired_h_sorted) // 2
for i in range(half):
    h1 = unpaired_h_sorted[i][1]
    h2 = unpaired_h_sorted[i + half][1]
    fallback_pairs_balanced.append((min(h1, h2), max(h1, h2)))

# --- Fallback 2: Optimal ---
fallback_pairs_optimal = []
sorted_all = sorted(unpaired_h, key=lambda x: x[1])
for i in range(len(sorted_all) // 2):
    h1 = sorted_all[i][1]
    h2 = sorted_all[-(i + 1)][1]
    fallback_pairs_optimal.append((min(h1, h2), max(h1, h2)))

# === Combine & Analyze
pairs_balanced_total = dl_pairs + fallback_pairs_balanced
pairs_optimal_total = dl_pairs + fallback_pairs_optimal

h_diffs_b, qos_b, rate_b = analyze_clusters(pairs_balanced_total)
h_diffs_o, qos_o, rate_o = analyze_clusters(pairs_optimal_total)

# === Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(sorted(h_diffs_b), 'o-', label=f"DL + Balanced | Rate: {rate_b:.2f} | QoS Fails: {qos_b}")
plt.plot(sorted(h_diffs_o), 's--', label=f"DL + Optimal | Rate: {rate_o:.2f} | QoS Fails: {qos_o}")
plt.axhline(y=qos_threshold, color='red', linestyle='--', label='QoS Threshold')
plt.title("Hybrid DL + Balanced vs DL + Optimal Pairing (Sorted Δh)")
plt.xlabel("Sorted Pair Index")
plt.ylabel("Δh = |h₂ - h₁|")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
