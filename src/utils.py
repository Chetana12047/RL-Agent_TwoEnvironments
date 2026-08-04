"""
utils.py

Utility functions for the Reinforcement Learning project.
"""

from pathlib import Path
import random

import numpy as np
import torch


# ==========================================================
# Set Random Seed
# ==========================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ==========================================================
# Create Directory
# ==========================================================

def create_directory(directory):

    Path(directory).mkdir(
        parents=True,
        exist_ok=True
    )


# ==========================================================
# Save Model
# ==========================================================

def save_model(model, file_path):

    torch.save(
        model.state_dict(),
        file_path
    )


# ==========================================================
# Load Model
# ==========================================================

def load_model(model, file_path, device):

    model.load_state_dict(
        torch.load(
            file_path,
            map_location=device
        )
    )

    model.eval()

    return model


# ==========================================================
# Calculate Average Reward
# ==========================================================

def average_reward(rewards):

    if len(rewards) == 0:

        return 0.0

    return float(np.mean(rewards))


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Testing Utility Functions")
    print("=" * 60)

    set_seed(42)

    rewards = [
        10,
        20,
        30,
        40,
        50
    ]

    print(
        "Average Reward :",
        average_reward(rewards)
    )

    create_directory("temp_test_folder")

    print("Directory created successfully.")

    print("\nUtility test completed successfully.")