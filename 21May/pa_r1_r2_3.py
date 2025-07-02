import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Parameters ===
np.random.seed(42)
N = 100
alpha = 3.5
P_total = 1.0
sigma_squared = 1e-9
qos_threshold = 0.0001

# === Generate channel gains
d = np.random.uniform(50, 500, N)
g = np.random.rayleigh(scale=1.0, size=N)
h = g / (d ** (alpha / 2))
sorted_indices = np.argsort(h)

# === Pairing
opt_pairs = [(min(h[sorted_indices[i]], h[sorted_indices[-(i + 1)]]),
              max(h[sorted_indices[i]], h[sorted_indices[-(i + 1)]]))
             for i in range(N // 2)]

bal_pairs = [(min(h[sorted_indices[i]], h[sorted_indices[i + N // 2]]),
              max(h[sorted_indices[i]], h[sorted_indices[i + N // 2]]))
             for i in range(N // 2)]

# === NOMA Rate Function
def noma_rate(P1, P2, h1, h2):
    R1 = np.log2(1 + (P1 * h1) / (P2 * h1 + sigma_squared))
    R2 = np.log2(1 + (P2 * h2) / sigma_squared)
    return R1, R2

# === Strategy Evaluation
def evaluate(pairs, strategy):
    R1s, R2s, total_rate, qos_fails = [], [], 0, 0
    for h1, h2 in pairs:
        if strategy == "convex":
            P1, P2 = 0.3 * P_total, 0.7 * P_total
        elif strategy == "heuristic":
            if h2 / h1 > 4:
                P1, P2 = 0.1 * P_total, 0.9 * P_total
            else:
                P1, P2 = 0.4 * P_total, 0.6 * P_total
        elif strategy == "learning":
            ratio = h1 / (h1 + h2)
            P1 = max(0.1, ratio * 0.5)
            P2 = P_total - P1
        R1, R2 = noma_rate(P1, P2, h1, h2)
        R1s.append(R1)
        R2s.append(R2)
        total_rate += R1 + R2
        if abs(h2 - h1) < qos_threshold:
            qos_fails += 1
    return {
        "Avg Rate": total_rate / len(pairs),
        "QoS Fails": qos_fails,
        "R1": R1s,
        "R2": R2s
    }

# === Run all 6 combinations
strategies = ["convex", "heuristic", "learning"]
results = {}
for strat in strategies:
    results[f"Opt-{strat}"] = evaluate(opt_pairs, strat)
    results[f"Bal-{strat}"] = evaluate(bal_pairs, strat)

# === Summary Table
summary_df = pd.DataFrame({
    name: {"Avg Rate": res["Avg Rate"], "QoS Fails": res["QoS Fails"]}
    for name, res in results.items()
}).T

# Print & Save
print("\n=== Summary of Power Allocation (3 Strategies × 2 Pairings) ===")
print(summary_df.to_string(index=True))
summary_df.to_csv("combined_power_allocation_summary.csv")

# === Combined Plot
plt.figure(figsize=(13, 6))
colors = {
    "convex": "blue",
    "heuristic": "green",
    "learning": "orange"
}
linestyles = {
    "Opt": "-",
    "Bal": "--"
}

for strat in strategies:
    for pairing in ["Opt", "Bal"]:
        r1 = results[f"{pairing}-{strat}"]["R1"]
        r2 = results[f"{pairing}-{strat}"]["R2"]
        label_r1 = f"{pairing} - {strat.capitalize()} R1"
        label_r2 = f"{pairing} - {strat.capitalize()} R2"
        plt.plot(r1, linestyle=linestyles[pairing], marker='o', label=label_r1, color=colors[strat], alpha=0.9)
        plt.plot(r2, linestyle=linestyles[pairing], marker='x', label=label_r2, color=colors[strat], alpha=0.5)

plt.xlabel("Pair Index")
plt.ylabel("Rate (bits/s/Hz)")
plt.title("R1 & R2 – Convex, Heuristic, Learning (Optimal vs Balanced Pairing)")
plt.grid(True)
plt.legend(loc="upper left", fontsize="small", ncol=2)
plt.tight_layout()
plt.savefig("combined_power_allocation_plot.png")
plt.show()
