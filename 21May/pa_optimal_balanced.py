import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === Parameters ===
np.random.seed(42)
N = 100
alpha = 3.5
P_total = 1.0
sigma_squared = 1e-9
qos_threshold = 0.0001

# === Simulate channel gains h ===
d = np.random.uniform(50, 500, N)
g = np.random.rayleigh(scale=1.0, size=N)
h = g / (d ** (alpha / 2))

# === Form Pairs ===

# Optimal: 1st with last, 2nd with second last...
sorted_indices = np.argsort(h)
optimal_pairs = [(min(h[sorted_indices[i]], h[sorted_indices[-(i + 1)]]),
                  max(h[sorted_indices[i]], h[sorted_indices[-(i + 1)]]))
                 for i in range(N // 2)]

# Balanced: 1st with 51st, 2nd with 52nd, ...
balanced_pairs = [(min(h[sorted_indices[i]], h[sorted_indices[i + N // 2]]),
                   max(h[sorted_indices[i]], h[sorted_indices[i + N // 2]]))
                  for i in range(N // 2)]

# === NOMA Rate Formula ===
def noma_rate(P1, P2, h1, h2):
    R1 = np.log2(1 + (P1 * h1) / (P2 * h1 + sigma_squared))
    R2 = np.log2(1 + (P2 * h2) / sigma_squared)
    return R1, R2

# === Power Allocation Evaluation Function ===
def evaluate_power_strategies(pairs):
    results = {}
    for strategy in ["static", "invh", "optimal"]:
        R1_list, R2_list = [], []
        qos_fails, total_rate = 0, 0
        for h1, h2 in pairs:
            if strategy == "static":
                P1, P2 = 0.2 * P_total, 0.8 * P_total
            elif strategy == "invh":
                P1 = (h2 / (h1 + h2)) * P_total
                P2 = (h1 / (h1 + h2)) * P_total
            elif strategy == "optimal":
                best_rate, best_alpha = 0, 0.5
                for alpha in np.linspace(0.01, 0.99, 99):
                    P1_temp = alpha * P_total
                    P2_temp = (1 - alpha) * P_total
                    R1, R2 = noma_rate(P1_temp, P2_temp, h1, h2)
                    if R1 + R2 > best_rate:
                        best_rate = R1 + R2
                        best_alpha = alpha
                P1 = best_alpha * P_total
                P2 = (1 - best_alpha) * P_total

            R1, R2 = noma_rate(P1, P2, h1, h2)
            R1_list.append(R1)
            R2_list.append(R2)
            total_rate += R1 + R2
            if abs(h2 - h1) < qos_threshold:
                qos_fails += 1

        avg_rate = total_rate / len(pairs)
        results[strategy] = {"Avg Rate": avg_rate, "QoS Fails": qos_fails, "R1s": R1_list, "R2s": R2_list}
    return results

# === Evaluate for both pairing methods ===
optimal_results = evaluate_power_strategies(optimal_pairs)
balanced_results = evaluate_power_strategies(balanced_pairs)

# === Plotting (Optimal Pairs Only) ===
labels = list(range(1, len(optimal_pairs) + 1))
plt.figure(figsize=(12, 6))
plt.plot(labels, optimal_results["static"]["R1s"], 'o-', label='Opt R1 Static')
plt.plot(labels, optimal_results["static"]["R2s"], 'o--', label='Opt R2 Static')
plt.plot(labels, optimal_results["invh"]["R1s"], 's-', label='Opt R1 Inverse-h')
plt.plot(labels, optimal_results["invh"]["R2s"], 's--', label='Opt R2 Inverse-h')
plt.plot(labels, optimal_results["optimal"]["R1s"], '^-', label='Opt R1 Optimal α')
plt.plot(labels, optimal_results["optimal"]["R2s"], '^--', label='Opt R2 Optimal α')
plt.title("Rate per User – Power Allocation on Optimal Pairing")
plt.xlabel("Pair Index")
plt.ylabel("Rate (bits/s/Hz)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("power_allocation_optimal_plot.png")
plt.show()

# === Summary Table ===
summary_data = {
    "Opt Static": optimal_results["static"],
    "Opt Inverse-h": optimal_results["invh"],
    "Opt Optimal α": optimal_results["optimal"],
    "Bal Static": balanced_results["static"],
    "Bal Inverse-h": balanced_results["invh"],
    "Bal Optimal α": balanced_results["optimal"]
}

summary_table = {
    k: {"Avg Rate": v["Avg Rate"], "QoS Fails": v["QoS Fails"]}
    for k, v in summary_data.items()
}
df_summary = pd.DataFrame(summary_table).T

print("\n=== Power Allocation: Optimal vs Balanced Pairing ===")
print(df_summary.to_string(index=True))

# Save results
df_summary.to_csv("power_allocation_summary.csv")
