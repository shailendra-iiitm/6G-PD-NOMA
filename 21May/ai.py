import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# === Parameters ===
N = 10000
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
autoencoder.fit(h_scaled, h_scaled, epochs=100, batch_size=10, verbose=0)

# === Extract latent features ===
features = encoder.predict(h_scaled)

# === KMeans on latent space ===
# === KMeans clustering in latent space ===
features = encoder.predict(h_scaled)
unique_feature_count = len(np.unique(features, axis=0))
k = min(4, unique_feature_count)

pairs = []

if k < 2:
    print("⚠️ Not enough variation in features to form clusters. Skipping pairing.")
else:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(features)

    # === Group users by cluster ===
    clusters = [[] for _ in range(k)]
    for i in range(N):
        clusters[labels[i]].append((i, h[i][0]))

    # === Sort clusters by avg h ===
    clusters = sorted(clusters, key=lambda c: np.mean([x[1] for x in c]))

    # === Pair formation: low-high cluster pairing ===
    used = set()
    i, j = 0, k - 1
    while i < j:
        low = sorted(clusters[i], key=lambda x: x[1])
        high = sorted(clusters[j], key=lambda x: x[1], reverse=True)
        for a, b in zip(low, high):
            if a[0] not in used and b[0] not in used:
                pairs.append((min(a[1], b[1]), max(a[1], b[1])))
                used.update([a[0], b[0]])
        i += 1
        j -= 1



# === Group users by cluster ===
clusters = [[] for _ in range(k)]
for i in range(N):
    clusters[labels[i]].append((i, h[i][0]))

# === Sort clusters by avg h ===
clusters = sorted(clusters, key=lambda c: np.mean([x[1] for x in c]))

# === Pair formation: low-high cluster pairing ===
pairs = []
used = set()
i, j = 0, k - 1
while i < j:
    low = sorted(clusters[i], key=lambda x: x[1])
    high = sorted(clusters[j], key=lambda x: x[1], reverse=True)
    for a, b in zip(low, high):
        if a[0] not in used and b[0] not in used:
            pairs.append((min(a[1], b[1]), max(a[1], b[1])))
            used.update([a[0], b[0]])
    i += 1
    j -= 1

# === NOMA rate function ===
def noma_rate(h1, h2):
    P1 = 0.8 * P_total
    P2 = 0.2 * P_total
    R1 = np.log2(1 + (P1 * h1) / (P2 * h1 + sigma_squared))
    R2 = np.log2(1 + (P2 * h2) / sigma_squared)
    return R1, R2

# === NOMA rate function (if missing, also re-add this)
def noma_rate(h1, h2):
    P1 = 0.8 * P_total
    P2 = 0.2 * P_total
    R1 = np.log2(1 + (P1 * h1) / (P2 * h1 + sigma_squared))
    R2 = np.log2(1 + (P2 * h2) / sigma_squared)
    return R1, R2

# === Cluster analysis function
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


# === Analyze pairs ===
# === Analyze only if we have pairs
if len(pairs) == 0:
    print("⚠️ No valid AI-based pairs formed. Try again with a different random seed or fewer clusters.")
else:
    h_diffs, qos_failures, avg_rate = analyze_clusters(pairs)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(h_diffs)+1), sorted(h_diffs), 'o-', color='purple', label='AI Clustering Δh')
    plt.axhline(y=qos_threshold, color='red', linestyle='--', label='QoS Threshold (0.0001)')
    plt.title("Sorted Channel Gain Differences - AI-Based Clustering")
    plt.xlabel("Sorted Pair Index")
    plt.ylabel("Δh = |h₂ - h₁|")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"✅ AI-Based Clustering Summary:")
    print(f"• Total Pairs        : {len(pairs)}")
    print(f"• Average Rate       : {avg_rate:.4f} bits/s/Hz")
    print(f"• QoS Failures (<0.0001): {qos_failures}")


# === Plot sorted h differences ===
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, len(h_diffs)+1), sorted(h_diffs), 'o-', color='purple', label='AI Clustering Δh')
plt.axhline(y=qos_threshold, color='red', linestyle='--', label='QoS Threshold (0.0001)')
plt.title("Sorted Channel Gain Differences - AI-Based Clustering")
plt.xlabel("Sorted Pair Index")
plt.ylabel("Δh = |h₂ - h₁|")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Print Results ===
print(f"✅ AI-Based Clustering Summary:")
print(f"• Total Pairs        : {len(pairs)}")
print(f"• Average Rate       : {avg_rate:.4f} bits/s/Hz")
print(f"• QoS Failures (<0.0001): {qos_failures}")
