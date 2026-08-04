"""
dqn.py

Deep Q-Network (DQN) model implemented using PyTorch.
"""

import torch
import torch.nn as nn

from config import DEVICE


class DQN(nn.Module):
    """
    Deep Q-Network for Reinforcement Learning.
    """

    def __init__(self, state_size, action_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(state_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_size)

        )

    # ======================================================
    # Forward Pass
    # ======================================================

    def forward(self, state):

        return self.network(state)


# ==========================================================
# Test Network
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Testing Deep Q-Network")
    print("=" * 60)

    state_size = 4
    action_size = 2

    model = DQN(state_size, action_size).to(DEVICE)

    sample_state = torch.randn((1, state_size)).to(DEVICE)

    output = model(sample_state)

    print("Device       :", DEVICE)
    print("Input Shape  :", sample_state.shape)
    print("Output Shape :", output.shape)
    print("Q Values     :", output)

    print("\nDQN test completed successfully.")