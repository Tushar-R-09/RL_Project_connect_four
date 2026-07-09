"""
agent.py: Deep Q-Network Brain and Replay Memory for Connect Four.
Features perspective inversion and explicit action masking for legal validation.
"""

import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque


class ConnectFourNet(nn.Module):
    def __init__(self, rows=6, cols=7):
        super().__init__()
        self.input_dim = rows * cols

        # A robust feed-forward multi-layer perceptron to map board vectors to Q-values
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, cols),  # 7 outputs matching the 7 columns
        )

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    def __init__(
        self, rows=6, cols=7, lr=1e-3, gamma=0.99, buffer_size=10000, batch_size=64
    ):
        self.cols = cols
        self.gamma = gamma
        self.batch_size = batch_size

        # Double-DQN Setup: Policy network and target stability network (Lecture 6)
        self.policy_net = ConnectFourNet(rows, cols)
        self.target_net = ConnectFourNet(rows, cols)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)

    def select_action(self, board, valid_actions, player_turn, epsilon=0.1):
        """Selects action using epsilon-greedy exploration mixed with action masking."""
        if len(valid_actions) == 0:
            return None

        # Exploration phase (Lecture 3 & 9)
        if random.random() < epsilon:
            return random.choice(valid_actions)

        # Exploitation phase: Pass the board to the network from the active player's perspective
        normalized_board = board.flatten() * player_turn
        state_tensor = torch.FloatTensor(normalized_board).unsqueeze(0)

        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze(0).numpy()

        # Action Masking: Force values of full columns to negative infinity
        masked_q_values = np.full(self.cols, -np.inf)
        masked_q_values[valid_actions] = q_values[valid_actions]

        return int(np.argmax(masked_q_values))

    def store_transition(self, state, action, reward, next_state, done, player_turn):
        """Stores the perspective-normalized transition inside the replay buffer."""
        # Normalize states so the acting player is always represented as +1
        norm_state = state.flatten() * player_turn
        norm_next_state = (
            next_state.flatten() * player_turn
            if next_state is not None
            else np.zeros_like(norm_state)
        )

        self.memory.append((norm_state, action, reward, norm_next_state, done))

    def train_step(self):
        """Performs a gradient descent update step across a batch of memory transitions."""
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(dones).unsqueeze(1)

        # Calculate current Q-values: Q(s, a)
        self.policy_net.train()
        current_q = self.policy_net(states_t).gather(1, actions_t)

        # Calculate target utility using the Target Network: R + gamma * max_a Q_target(s', a)
        with torch.no_grad():
            # Invert perspective for next_state because the opponent acts next!
            max_next_q = self.target_net(-next_states_t).max(1)[0].unsqueeze(1)
            # Standard Bellman equation target mapping
            target_q = rewards_t + (self.gamma * max_next_q * (1 - dones_t))

        # Compute Mean Squared Error Loss and optimize parameters
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Syncs weights from policy network to target network for stability."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
