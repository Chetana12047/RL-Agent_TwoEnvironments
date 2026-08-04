"""
config.py

Central configuration for the Reinforcement Learning project.
"""

from pathlib import Path
import torch

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ==========================================================
# Device
# ==========================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ==========================================================
# Environment Configuration
# ==========================================================

ENVIRONMENTS = [
    "CartPole-v1",
    "FrozenLake-v1",
]

# ==========================================================
# Training Hyperparameters
# ==========================================================

EPISODES = 500

MAX_STEPS = 500

BATCH_SIZE = 64

LEARNING_RATE = 0.001

GAMMA = 0.99

EPSILON_START = 1.0

EPSILON_END = 0.01

EPSILON_DECAY = 0.995

TARGET_UPDATE = 10

MEMORY_SIZE = 10000

# ==========================================================
# Random Seed
# ==========================================================

SEED = 42