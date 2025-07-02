import numpy as np
import matplotlib.pyplot as plt

# fixed params
h1, h2 = 0.3, 1.0
P, N0   = 1.0, 1.0

# OMA sum-rate (constant)
R1_oma = 0.5 * np.log2(1 + P * h1 / N0)
R2_oma = 0.5 * np.log2(1 + P * h2 / N0)
sum_oma = R1_oma + R2_oma

# sweep a1 from 0.1 to 0.9
a1_vals = np.linspace(0.1, 0.9, 81)
sum_noma = []
for a1 in a1_vals:
    a2 = 1 - a1
    R1 = np.log2(1 + (a1 * P * h1) / (a2 * P * h1 + N0))
    R2 = np.log2(1 + (a2 * P * h2) / N0)
    sum_noma.append(R1 + R2)

# plot
plt.figure(figsize=(7,4))
plt.plot(a1_vals, sum_noma, label='NOMA Sum-Rate')
plt.hlines(sum_oma, a1_vals[0], a1_vals[-1],
           colors='gray', linestyles='--', label='OMA Sum-Rate')
plt.xlabel('Power Split to Weak User, a₁')
plt.ylabel('Sum‑Rate (bps/Hz)')
plt.title('NOMA vs OMA: Optimal Power Allocation for 2 Users')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
