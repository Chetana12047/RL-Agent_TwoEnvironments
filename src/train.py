"""
train.py

Training script for Deep Q-Network (DQN)
on multiple Gymnasium environments.
"""

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from config import (
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    ENVIRONMENTS,
    EPISODES,
    MAX_STEPS,
)

from environment import EnvironmentManager
from agent import DQNAgent


# ==========================================================
# Train One Environment
# ==========================================================

def train_environment(environment_name):

    print("=" * 60)
    print(f"Training Environment : {environment_name}")
    print("=" * 60)

    env = EnvironmentManager(environment_name)

    state_size = env.get_state_size()
    action_size = env.get_action_size()

    agent = DQNAgent(
        state_size,
        action_size
    )

    rewards_history = []

    best_reward = float("-inf")

    checkpoint_dir = Path(CHECKPOINT_DIR)

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_dir = Path(OUTPUT_DIR) / environment_name

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    # ======================================================
    # Training Loop
    # ======================================================

    for episode in tqdm(range(1, EPISODES + 1)):

        state = env.reset()

        if np.isscalar(state):

            state = np.eye(state_size)[int(state)]

        total_reward = 0

        for step in range(MAX_STEPS):

            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)

            if np.isscalar(next_state):

                next_state = np.eye(state_size)[int(next_state)]

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            agent.replay()

            state = next_state

            total_reward += reward

            if done:

                break

        rewards_history.append(total_reward)

        if total_reward > best_reward:

            best_reward = total_reward

            torch.save(
                agent.model.state_dict(),
                checkpoint_dir / f"{environment_name}_best_model.pth"
            )

        if episode % 20 == 0:

            average_reward = np.mean(
                rewards_history[-20:]
            )

            print(
                f"Episode {episode:4d} | "
                f"Average Reward : {average_reward:.2f} | "
                f"Epsilon : {agent.epsilon:.3f}"
            )
    env.close()

    # ======================================================
    # Save Reward Curve
    # ======================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        rewards_history,
        label="Episode Reward"
    )

    plt.title(
        f"Training Reward - {environment_name}"
    )

    plt.xlabel("Episode")

    plt.ylabel("Reward")

    plt.grid(True)

    plt.legend()

    plt.savefig(
        output_dir / "training_rewards.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    average_reward = np.mean(rewards_history)

    print("\nTraining Finished!")
    print(f"Best Reward    : {best_reward:.2f}")
    print(f"Average Reward : {average_reward:.2f}")

    return {
        "environment": environment_name,
        "best_reward": best_reward,
        "average_reward": average_reward,
        "episodes": EPISODES
    }


# ==========================================================
# Main
# ==========================================================

def main():

    random.seed(42)

    np.random.seed(42)

    torch.manual_seed(42)

    results = []

    print("=" * 60)
    print("Deep Q-Network Training")
    print("=" * 60)

    for environment_name in ENVIRONMENTS:

        result = train_environment(environment_name)

        results.append(result)

    print("\n")

    print("=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    for result in results:

        print(f"\nEnvironment    : {result['environment']}")
        print(f"Episodes       : {result['episodes']}")
        print(f"Best Reward    : {result['best_reward']:.2f}")
        print(f"Average Reward : {result['average_reward']:.2f}")

    print("\nTraining completed successfully!")


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":

    main()
