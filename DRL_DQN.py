import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Load dataset from C++ simulation
data = pd.read_csv("kmeans_data.csv")

# Define State and Action Space
state_columns = ["User1_ChannelGain", "User2_ChannelGain"]
num_actions = 10  # Discrete power allocation choices (0.1 to 0.9)
action_values = np.linspace(0.1, 0.9, num_actions)  # Possible power allocations

# Convert dataset into states
states = data[state_columns].values

# Define Deep Q-Network (DQN)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(input_dim=len(state_columns), output_dim=num_actions).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Experience Replay Memory
replay_memory = deque(maxlen=10000)
batch_size = 64
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration probability

# Initialize loss variable
loss = None  

# Training Loop
for episode in range(1000):
    # Select a random user pair
    idx = random.randint(0, len(states) - 2)  # Ensure next state exists
    state = torch.tensor(states[idx], dtype=torch.float32).to(device)
    next_state = torch.tensor(states[idx + 1], dtype=torch.float32).to(device)

    # Choose action (power allocation)
    if random.random() < epsilon:  # Exploration
        action_idx = random.randint(0, num_actions - 1)
    else:  # Exploitation
        with torch.no_grad():
            q_values = model(state.unsqueeze(0))  # Add batch dimension
            action_idx = torch.argmax(q_values).item()

    # Compute reward
    P_s = action_values[action_idx]
    P_w = 1 - P_s
    h_s, h_w = states[idx]
    SINR_s = P_s * h_s / (0.1 + P_w * h_w)  # Simplified SINR formula
    SINR_w = P_w * h_w / (0.1 + P_s * h_s)
    reward = np.log2(1 + SINR_s) + np.log2(1 + SINR_w)

    # Store experience in replay buffer
    done = idx == len(states) - 2  # Check if last state
    replay_memory.append((state, action_idx, reward, next_state, done))

    # Sample batch and train
    if len(replay_memory) > batch_size:
        batch = random.sample(replay_memory, batch_size)
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*batch)

        batch_states = torch.stack(batch_states).to(device)
        batch_actions = torch.tensor(batch_actions, dtype=torch.int64).to(device)
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(device)
        batch_next_states = torch.stack(batch_next_states).to(device)
        batch_dones = torch.tensor(batch_dones, dtype=torch.float32).to(device)

        q_values = model(batch_states)
        next_q_values = model(batch_next_states).detach()  # Detach next state Q-values

        # Compute target Q-values
        target_q_values = q_values.clone()
        for i in range(batch_size):
            target_q_values[i, batch_actions[i]] = batch_rewards[i] + gamma * torch.max(next_q_values[i]) * (1 - batch_dones[i])

        # Compute loss and optimize
        loss = criterion(q_values, target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ✅ Print loss only if it was computed
    if episode % 100 == 0 and loss is not None:
        print(f"Episode {episode}, Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), "drl_power_allocation.pth")
print("Model trained and saved as drl_power_allocation.pth")
import onnx

# Load trained model
model.eval()  # Set to evaluation mode

# Define input tensor (Assuming input has 2 features: channel gains)
dummy_input = torch.randn(1, 2)  # Adjust shape based on your input size

# Convert to ONNX
onnx_file = "drl_power_allocation.onnx"
torch.onnx.export(model, dummy_input, onnx_file, export_params=True, opset_version=11, input_names=['input'], output_names=['output'])

print(f"✅ Model converted and saved as {onnx_file}")
