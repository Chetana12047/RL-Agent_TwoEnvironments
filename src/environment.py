"""
environment.py

Handles creation and management of reinforcement learning environments.
"""

import gymnasium as gym

from config import SEED


class EnvironmentManager:
    """
    Creates and manages Gymnasium environments.
    """

    def __init__(self, environment_name):

        self.environment_name = environment_name

        self.environment = gym.make(environment_name)

        self.state, _ = self.environment.reset(seed=SEED)

    # ======================================================
    # Reset Environment
    # ======================================================

    def reset(self):

        self.state, _ = self.environment.reset(seed=SEED)

        return self.state

    # ======================================================
    # Take One Step
    # ======================================================

    def step(self, action):

        next_state, reward, terminated, truncated, info = self.environment.step(action)

        done = terminated or truncated

        return next_state, reward, done, info

    # ======================================================
    # State Size
    # ======================================================

    def get_state_size(self):

        observation_space = self.environment.observation_space

        if hasattr(observation_space, "n"):

            return observation_space.n

        return observation_space.shape[0]

    # ======================================================
    # Action Size
    # ======================================================

    def get_action_size(self):

        return self.environment.action_space.n

    # ======================================================
    # Close Environment
    # ======================================================

    def close(self):

        self.environment.close()


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Testing Environment Manager")
    print("=" * 60)

    for env_name in ["CartPole-v1", "FrozenLake-v1"]:

        print(f"\nEnvironment : {env_name}")

        manager = EnvironmentManager(env_name)

        print("State Size :", manager.get_state_size())
        print("Action Size:", manager.get_action_size())

        state = manager.reset()

        print("Initial State:", state)

        manager.close()

    print("\nEnvironment test completed successfully.")