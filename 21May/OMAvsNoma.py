import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_users = 5  # Number of users in the system
snr_db = np.arange(0, 31, 5)  # SNR range in dB
bandwidth = 1e6  # 1 MHz bandwidth
noise_power = 1e-12  # Noise power in Watts

# OMA (TDMA) capacity calculation
def oma_capacity(snr_linear, num_users):
    # Equal time/frequency allocation
    return bandwidth * np.log2(1 + snr_linear) / num_users

# NOMA capacity calculation
def noma_capacity(snr_linear, num_users):
    # Assuming perfect SIC and ordered power allocation
    # Users are ordered with decreasing channel gains
    user_capacities = []
    for i in range(1, num_users + 1):
        # Power allocation coefficients (alpha_i)
        alpha = 1 / (num_users - i + 1)  # Simple power allocation
        interference = sum([1/(num_users - k + 1) for k in range(i + 1, num_users + 1)])
        sinr = (alpha * snr_linear) / (interference * snr_linear + 1)
        user_capacities.append(bandwidth * np.log2(1 + sinr))
    
    return sum(user_capacities)

# Convert SNR from dB to linear
snr_linear = 10 ** (snr_db / 10)

# Calculate capacities
oma_sum_rate = []
noma_sum_rate = []

for snr in snr_linear:
    oma_sum_rate.append(oma_capacity(snr, num_users) * num_users / 1e6)  # Convert to Mbps
    noma_sum_rate.append(noma_capacity(snr, num_users) / 1e6)  # Convert to Mbps

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(snr_db, oma_sum_rate, 'b-o', linewidth=2, label='OMA (TDMA)')
plt.plot(snr_db, noma_sum_rate, 'r--s', linewidth=2, label='NOMA')
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Sum Rate (Mbps)', fontsize=12)
plt.title('Comparison of OMA and NOMA Systems', fontsize=14)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.xticks(snr_db)
plt.tight_layout()
plt.show()