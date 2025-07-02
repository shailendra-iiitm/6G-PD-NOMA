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

# === Train autoencoder ===
autoencoder.fit(h_scaled, h_scaled, epochs=100, batch_size=256, verbose=0)

# === Utility Functions ===
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

def match_clusters(c1, c2):
    used = set()
    pairs = []
    low = sorted(c1, key=lambda x: x[1])
    high = sorted(c2, key=lambda x: x[1], reverse=True)
    for a, b in zip(low, high):
        if a[0] not in used and b[0] not in used:
            pairs.append((min(a[1], b[1]), max(a[1], b[1])))
            used.update([a[0], b[0]])
    return pairs

# === Try until 4 clusters are formed
results = {}
max_attempts = 10
for attempt in range(max_attempts):
    seed = np.random.randint(1000)
    kmeans = KMeans(n_clusters=4, random_state=seed)
    features = encoder.predict(h_scaled)
    labels = kmeans.fit_predict(features)
    unique_clusters = len(set(labels))
    if unique_clusters == 4:
        print(f"✅ Clustering succeeded on attempt {attempt+1} with seed {seed}")
        break
else:
    print("❌ Failed to get 4 clusters even after several attempts.")
    exit()

# === Proceed with pairing only if 4 clusters formed
clusters = [[] for _ in range(4)]
for i in range(N):
    clusters[labels[i]].append((i, h[i][0]))

clusters = sorted(clusters, key=lambda c: np.mean([x[1] for x in c]))

# Strategy 1: Cluster 0 <-> 2 and 1 <-> 3
pairs_1 = match_clusters(clusters[0], clusters[2]) + match_clusters(clusters[1], clusters[3])
h_diffs_1, qos_failures_1, avg_rate_1 = analyze_clusters(pairs_1)
results['0<->2 & 1<->3'] = (h_diffs_1, avg_rate_1, qos_failures_1)

# Strategy 2: Cluster 0 <-> 3 and 1 <-> 2
pairs_2 = match_clusters(clusters[0], clusters[3]) + match_clusters(clusters[1], clusters[2])
h_diffs_2, qos_failures_2, avg_rate_2 = analyze_clusters(pairs_2)
results['0<->3 & 1<->2'] = (h_diffs_2, avg_rate_2, qos_failures_2)

# === Plotting both strategies (limit max pairs for visibility if large)
plt.figure(figsize=(12, 6))
for label, (diffs, rate, fails) in results.items():
    subset = sorted(diffs)[:500]  # Only show first 500 points
    plt.plot(subset, 'o-', label=f"{label} | Rate: {rate:.2f} | QoS fails: {fails}")
plt.axhline(y=qos_threshold, color='red', linestyle='--', label='QoS Threshold (0.0001)')
plt.title("Cluster Matching Strategies Comparison (Sorted Δh)")
plt.xlabel("Sorted Pair Index")
plt.ylabel("Δh = |h₂ - h₁|")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
