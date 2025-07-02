import numpy as np
import time
import matplotlib.pyplot as plt

def generate_channel_gains(n):
    """Generate Rayleigh fading channel gains between 0 and 1."""
    gains = np.random.rayleigh(scale=0.3, size=n)  # Adjust scale to keep values in [0,1]
    return np.sort(np.clip(gains, 0, 1))

def centroid_based_pairing(h):
    """My method: Pair users around the median (centroid)."""
    n = len(h)
    pairs = [(h[i], h[n - i - 1]) for i in range(n // 2)]
    return pairs

def max_difference_pairing(h):
    """Your method: Pair users with maximum difference."""
    n = len(h)
    pairs = [(h[i], h[i + (n // 2)]) for i in range(n // 2)]
    return pairs

def evaluate_pairing(pairs, threshold):
    """Compute average difference, standard deviation, and SIC success rate."""
    differences = [abs(u1 - u2) for u1, u2 in pairs]
    sic_failures = sum(1 for d in differences if d < threshold)
    return np.mean(differences), np.std(differences), sic_failures, pairs, differences

# Number of users
n = 10
h_threshold = 0.5  # Threshold for successful SIC

# Generate random channel gains
h = generate_channel_gains(n)
print("Generated channel gains:", h)

# Histogram of channel gains
plt.figure()
plt.hist(h, bins=10, color='skyblue', edgecolor='black')
plt.xlabel("Channel Gain Values")
plt.ylabel("Frequency")
plt.title("Histogram of Generated Channel Gains")
plt.show()

# Measure execution time and apply both methods
start = time.time()
my_pairs = centroid_based_pairing(h)
my_time = time.time() - start

start = time.time()
your_pairs = max_difference_pairing(h)
your_time = time.time() - start

# Evaluate both methods
my_avg_diff, my_std, my_failures, my_pair_list, my_differences = evaluate_pairing(my_pairs, h_threshold)
your_avg_diff, your_std, your_failures, your_pair_list, your_differences = evaluate_pairing(your_pairs, h_threshold)

# Print comparison
print("Method\t\tAvg Difference\tStd Dev\tSIC Failures\tExecution Time")
print(f"My Method\t{my_avg_diff:.4f}\t{my_std:.4f}\t{my_failures}\t{my_time:.6f} sec")
print(f"Your Method\t{your_avg_diff:.4f}\t{your_std:.4f}\t{your_failures}\t{your_time:.6f} sec")

# Print pairs and their differences
print("\nMy Method Pairs and Differences:")
for pair, diff in zip(my_pair_list, my_differences):
    print(f"Pair: {pair}, Difference: {diff:.4f}")

print("\nYour Method Pairs and Differences:")
for pair, diff in zip(your_pair_list, your_differences):
    print(f"Pair: {pair}, Difference: {diff:.4f}")

# Visualization
x_labels = ['My Method', 'Your Method']
values = [my_avg_diff, your_avg_diff]
std_dev = [my_std, your_std]
failures = [my_failures, your_failures]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(x_labels, values, color=['blue', 'green'], alpha=0.6, label='Avg Difference')
ax2.plot(x_labels, std_dev, color='red', marker='o', label='Std Dev')
ax2.plot(x_labels, failures, color='black', marker='s', label='SIC Failures')

ax1.set_xlabel("Pairing Method")
ax1.set_ylabel("Average Difference", color='blue')
ax2.set_ylabel("Std Dev / SIC Failures", color='red')
plt.title("Comparison of Pairing Methods with SIC Constraint")
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()