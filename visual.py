import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the simulation results
df = pd.read_csv("6G_simulation_data.csv")

# Plot channel gain distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["User1_ChannelGain"], bins=50, color="blue", label="Strong Users", kde=True)
sns.histplot(df["User2_ChannelGain"], bins=50, color="red", label="Weak Users", kde=True)
plt.xlabel("Channel Gain")
plt.ylabel("Frequency")
plt.title("Channel Gain Distribution of Strong & Weak Users")
plt.legend()
plt.show()

# Plot power allocation distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["Power1"], bins=10, color="green", label="Power to Strong User", kde=True)
sns.histplot(df["Power2"], bins=10, color="orange", label="Power to Weak User", kde=True)
plt.xlabel("Power Allocation")
plt.ylabel("Frequency")
plt.title("DRL-Based Power Allocation Distribution")
plt.legend()
plt.show()

# Scatter plot of power allocation vs channel gain
plt.figure(figsize=(10, 5))
plt.scatter(df["User1_ChannelGain"], df["Power1"], color="blue", label="Strong User Power", alpha=0.5)
plt.scatter(df["User2_ChannelGain"], df["Power2"], color="red", label="Weak User Power", alpha=0.5)
plt.xlabel("Channel Gain")
plt.ylabel("Power Allocation")
plt.title("Power Allocation vs. Channel Gain")
plt.legend()
plt.show()

print("✅ Visualization Complete. Check the plots above!")
