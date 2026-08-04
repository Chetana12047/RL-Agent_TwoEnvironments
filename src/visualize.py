"""
visualize.py

Visualize training reward curves for reinforcement learning.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from config import OUTPUT_DIR, ENVIRONMENTS


def visualize_environment(environment_name):

    image_path = (
        Path(OUTPUT_DIR)
        / environment_name
        / "training_rewards.png"
    )

    if not image_path.exists():

        print(f"Reward plot not found for {environment_name}")

        return

    image = mpimg.imread(image_path)

    plt.figure(figsize=(10, 5))

    plt.imshow(image)

    plt.axis("off")

    plt.title(f"{environment_name} Reward Curve")

    plt.show()


def main():

    print("=" * 60)
    print("Training Reward Visualization")
    print("=" * 60)

    for environment_name in ENVIRONMENTS:

        visualize_environment(environment_name)

    print("\nVisualization completed successfully.")


if __name__ == "__main__":

    main()