"""
evaluate.py

Evaluate trained Deep Q-Network models.
"""

from pathlib import Path

import numpy as np
import torch

from config import (
    CHECKPOINT_DIR,
    ENVIRONMENTS,
    DEVICE,
)

from environment import EnvironmentManager
from dqn import DQN


# ==========================================================
# Evaluate One Environment
# ==========================================================

def evaluate_environment(environment_name):

    print("=" * 60)
    print(f"Evaluating : {environment_name}")
    print("=" * 60)

    env = EnvironmentManager(environment_name)

    state_size = env.get_state_size()

    action_size = env.get_action_size()

    model = DQN(
        state_size,
        action_size
    ).to(DEVICE)

    checkpoint = (
        Path(CHECKPOINT_DIR)
        /
        f"{environment_name}_best_model.pth"
    )

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=DEVICE
        )
    )

    model.eval()

    rewards = []
    # ======================================================
    # Evaluation Loop
    # ======================================================

    for episode in range(20):

        state = env.reset()

        if np.isscalar(state):

            state = np.eye(state_size)[int(state)]

        total_reward = 0

        done = False

        while not done:

            state_tensor = (
                torch.FloatTensor(state)
                .unsqueeze(0)
                .to(DEVICE)
            )

            with torch.no_grad():

                q_values = model(state_tensor)

                action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)

            if np.isscalar(next_state):

                next_state = np.eye(state_size)[int(next_state)]

            state = next_state

            total_reward += reward

        rewards.append(total_reward)

    env.close()

    average_reward = np.mean(rewards)

    best_reward = np.max(rewards)

    print(f"Average Reward : {average_reward:.2f}")

    print(f"Best Reward    : {best_reward:.2f}")

    return {

        "environment": environment_name,

        "average_reward": average_reward,

        "best_reward": best_reward

    }
# ==========================================================
# Main
# ==========================================================

def main():

    results = []

    print("=" * 60)
    print("Deep Q-Network Evaluation")
    print("=" * 60)

    for environment_name in ENVIRONMENTS:

        result = evaluate_environment(
            environment_name
        )

        results.append(result)

    print("\n")

    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for result in results:

        print(f"\nEnvironment    : {result['environment']}")
        print(f"Average Reward : {result['average_reward']:.2f}")
        print(f"Best Reward    : {result['best_reward']:.2f}")

    print("\nEvaluation completed successfully!")


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":

    main()