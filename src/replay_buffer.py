"""
replay_buffer.py

Experience Replay Buffer for Deep Q-Network.
"""

from collections import deque
import random

import numpy as np
import torch

from config import DEVICE


class ReplayBuffer:
    """
    Stores experiences for training the DQN.
    """

    def __init__(self, capacity):

        self.memory = deque(maxlen=capacity)

    # ======================================================
    # Store Experience
    # ======================================================

    def push(self, state, action, reward, next_state, done):

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    # ======================================================
    # Sample Mini Batch
    # ======================================================

    def sample(self, batch_size):

        batch = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(DEVICE)

        actions = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)

        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)

        next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)

        dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    # ======================================================
    # Buffer Size
    # ======================================================

    def __len__(self):

        return len(self.memory)


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Testing Replay Buffer")
    print("=" * 60)

    buffer = ReplayBuffer(capacity=100)

    for _ in range(20):

        state = np.random.rand(4)

        action = random.randint(0, 1)

        reward = random.random()

        next_state = np.random.rand(4)

        done = random.choice([True, False])

        buffer.push(
            state,
            action,
            reward,
            next_state,
            done
        )

    print("Stored Experiences :", len(buffer))

    batch = buffer.sample(8)

    print("Batch Size :", batch[0].shape)

    print("\nReplay Buffer test completed successfully.")