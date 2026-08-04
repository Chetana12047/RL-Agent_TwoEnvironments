"""
agent.py

Deep Q-Network Agent.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    DEVICE,
    LEARNING_RATE,
    GAMMA,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY,
    MEMORY_SIZE,
    BATCH_SIZE,
)

from dqn import DQN
from replay_buffer import ReplayBuffer


class DQNAgent:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.epsilon = EPSILON_START

        self.memory = ReplayBuffer(MEMORY_SIZE)

        self.model = DQN(state_size, action_size).to(DEVICE)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE
        )

        self.loss_function = nn.MSELoss()

    # ======================================================
    # Choose Action
    # ======================================================

    def act(self, state):

        if random.random() < self.epsilon:

            return random.randrange(self.action_size)

        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            q_values = self.model(state)

        return torch.argmax(q_values).item()

    # ======================================================
    # Store Experience
    # ======================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )

    # ======================================================
    # Train
    # ======================================================

    def replay(self):

        if len(self.memory) < BATCH_SIZE:

            return

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = self.memory.sample(BATCH_SIZE)

        current_q = self.model(states).gather(1, actions)

        with torch.no_grad():

            next_q = self.model(next_states).max(1)[0].unsqueeze(1)

            target_q = rewards + GAMMA * next_q * (1 - dones)

        loss = self.loss_function(current_q, target_q)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        if self.epsilon > EPSILON_END:

            self.epsilon *= EPSILON_DECAY


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Testing DQN Agent")
    print("=" * 60)

    state_size = 4

    action_size = 2

    agent = DQNAgent(
        state_size,
        action_size
    )

    dummy_state = np.random.rand(state_size)

    action = agent.act(dummy_state)

    print("Selected Action :", action)

    print("Initial Epsilon :", agent.epsilon)

    print("\nDQN Agent test completed successfully.")