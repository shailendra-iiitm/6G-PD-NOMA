import numpy as np
import time
import matplotlib.pyplot as plt

def generate_channel_gains(n):
    """Generate Rayleigh fading channel gains."""
    return np.sort(np.random.rayleigh(scale=1.0, size=n))

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

def evaluate_pairing(pairs):
    """Compute average difference and standard deviation of pairwise gains."""
    differences = [abs(u1 - u2) for u1, u2 in pairs]
    return np.mean(differences), np.std(differences)

# Number of users
n = 10

# Generate random channel gains
h = generate_channel_gains(n)

# Measure execution time and apply both methods
start = time.time()
my_pairs = centroid_based_pairing(h)
my_time = time.time() - start

start = time.time()
your_pairs = max_difference_pairing(h)
your_time = time.time() - start

# Evaluate both methods
my_avg_diff, my_std = evaluate_pairing(my_pairs)
your_avg_diff, your_std = evaluate_pairing(your_pairs)

# Print comparison
print("Method\t\tAvg Difference\tStd Dev\tExecution Time")
print(f"My Method\t{my_avg_diff:.4f}\t{my_std:.4f}\t{my_time:.6f} sec")
print(f"Your Method\t{your_avg_diff:.4f}\t{your_std:.4f}\t{your_time:.6f} sec")

# Visualization
x_labels = ['My Method', 'Your Method']
values = [my_avg_diff, your_avg_diff]
plt.bar(x_labels, values, color=['blue', 'green'])
plt.xlabel("Pairing Method")
plt.ylabel("Average Difference")
plt.title("Comparison of Pairing Methods")
plt.show()
