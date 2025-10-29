import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path

def plot_experiment_results(experiment_path: Path, testing_period: int):
    """
    Loads and plots training and testing rewards from a UDRL experiment directory.
    """
    print(f"Loading results from: {experiment_path}")

    # --- Load Reward Data ---
    train_rewards_path = experiment_path / "train_rewards.npy"
    test_rewards_path = experiment_path / "test_rewards.npy"

    if not train_rewards_path.exists():
        print(f"Error: train_rewards.npy not found in {experiment_path}")
        return
    if not test_rewards_path.exists():
         print(f"Error: test_rewards.npy not found in {experiment_path}")
         return

    train_rewards = np.load(train_rewards_path)
    test_rewards = np.load(test_rewards_path)

    # Check if files are empty
    if train_rewards.size == 0:
        print("Warning: train_rewards.npy is empty. Did you remember to fix the run_experiment.py script?")
        return
    if test_rewards.size == 0:
        print("Warning: test_rewards.npy is empty. Skipping test plot.")

    # Separate mean and standard deviation
    train_mean = train_rewards[:, 0]
    train_std = train_rewards[:, 1]
    
    # --- Create X-axis ---
    # Training data is saved every episode
    x_train = np.arange(len(train_mean))
    
    # --- Generate Plot ---
    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    # Plot Training Rewards
    ax.plot(x_train, train_mean, label="Training Reward (Exploratory)")
    ax.fill_between(x_train, train_mean - train_std, train_mean + train_std, alpha=0.2,
                    label="Training Reward (Std Dev)")

    # Plot Testing Rewards (if it's not empty)
    if test_rewards.size > 0:
        test_mean = test_rewards[:, 0]
        test_std = test_rewards[:, 1]
        
        # Testing data is saved every 'testing_period' episodes
        x_test = (np.arange(len(test_mean)) + 1) * testing_period
        
        ax.plot(x_test, test_mean, label="Testing Reward (Fixed Goal)", marker='o', linestyle='--')
        ax.fill_between(x_test, test_mean - test_std, test_mean + test_std, alpha=0.2,
                        label="Testing Reward (Std Dev)")

    ax.set_title(f"UDRL Performance on {experiment_path.parent.name}") # Use folder name for title
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.legend()
    ax.grid(True)

    # Save the plot
    save_path = experiment_path / "performance_curve.png"
    plt.savefig(save_path)
    print(f"\nPlot saved successfully to: {save_path}")
    # plt.show() # Uncomment this if you want to display the plot immediately

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot UDRL experiment results.")
    parser.add_argument(
        "--path", 
        type=str, 
        required=True, 
        help="Path to the specific experiment directory (e.g., 'data/SimGlucoseEnv.../42/')"
    )
    parser.add_argument(
        "--testing_period", 
        type=int, 
        required=True, 
        help="The 'testing_period' value used during the experiment (e.g., 50)"
    )
    args = parser.parse_args()
    
    plot_experiment_results(Path(args.path), args.testing_period)