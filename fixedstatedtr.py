import gymnasium as gym
import torch
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm, trange
import imageio
from lime import lime_tabular
from pathlib import Path
from itertools import zip_longest
from dataclasses import dataclass, asdict

from udrl.cli import apply

# Import from your DTR/UDRL project
from dtrimpl import UDRLExperiment, create_argparse_dict, create_experiment_from_args, dataclass_non_defaults_to_string
from udrl.policies import NeuralPolicy, SklearnPolicy
from udrl.agent import UpsideDownAgent, AgentHyper

# ==================================
# 1. Explainer and Plotting Functions
# (These remain the same as your correct version)
# ==================================
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def predict_fn_nn(inputs, model, num_state_features):
    """
    FIXED prediction function for NeuralPolicy.
    Applies softmax to convert logits to probabilities.
    """
    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, axis=0)
    state = torch.tensor(inputs[:, :num_state_features], dtype=torch.float32)
    command = torch.tensor(inputs[:, num_state_features:], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(state, command)
        probs = torch.nn.functional.softmax(logits, dim=1).numpy()
    return probs

def predict_fn_sklearn(inputs, model, num_state_features):
    """ Prediction function for SklearnPolicy. """
    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, axis=0)
    probs = model.predict_proba(inputs)
    return probs

def summarize_explanations(episodes_explanations, summarize_funs=[np.mean, np.std]):
    """Averages explanations across multiple episodes for each time step."""
    transposed_data = list(zip_longest(*episodes_explanations, fillvalue=None))
    summary = []
    for time_step_data in transposed_data:
        valid_data = [d for d in time_step_data if d is not None]
        if not valid_data:
            summary.append([])
            continue
        features_at_timestep = list(zip(*valid_data))
        feature_summary = []
        for feature_data in features_at_timestep:
            proc_data = np.abs(np.array(feature_data))
            feature_summary.append([fun(proc_data) for fun in summarize_funs])
        summary.append(feature_summary)
    return summary

def plot_explanations_and_create_gifs(explanations, plot_dir, title_prefix, feature_names):
    """ Plots the average explanations and creates a GIF. """
    plot_dir.mkdir(parents=True, exist_ok=True)
    image_filenames = []
    for idx, feat_data in tqdm(
        enumerate(explanations), desc=f"Plotting {title_prefix}", total=len(explanations)
    ):
        if not feat_data: continue
        x_pos = np.arange(len(feat_data))
        plt.figure(figsize=(10, 6))
        plt.title(f"{title_prefix} Explanations at Time Step {idx}")
        means = [x[0] for x in feat_data]
        stds = [x[1] for x in feat_data]
        plt.bar(x_pos, means, yerr=stds, capsize=5)
        plt.xticks(x_pos, feature_names, rotation=45, ha="right")
        plt.xlabel("Features")
        plt.ylabel(f"Mean Absolute {title_prefix} Value")
        plt.tight_layout()
        filepath = plot_dir / f"{title_prefix.lower()}_timestep_{idx}.png"
        plt.savefig(filepath)
        plt.close()
        image_filenames.append(filepath)

    gif_path = plot_dir.parent / f"{title_prefix.lower()}_fixed_state_animation.gif" # Added suffix
    if image_filenames:
        print(f"Creating GIF for {title_prefix}...")
        image_filenames.sort(key=lambda f: int(f.stem.split('_')[-1]))
        images = [imageio.imread(f) for f in image_filenames]
        imageio.mimsave(gif_path, images, duration=0.5)
        print(f"Successfully saved {title_prefix} GIF to {gif_path}")

# ==================================
# 2. Main Script Execution
# ==================================

if __name__ == "__main__":
    # --- 1. Configuration (DYNAMIC) ---
    parser = argparse.ArgumentParser(description="Runs XAI analysis on a trained DTR-UDRL agent.")
    arguments = create_argparse_dict(UDRLExperiment)

    parser.add_argument("--num-simulations", type=int, default=10, help="Number of test episodes for XAI.")
    parser.add_argument("--use-fixed-goal", action="store_true", help="Use fixed goals instead of dynamic ones.")
    # --- NEW ARGUMENT ---
    parser.add_argument("--use-fixed-state", action="store_true", help="Run all simulations from the same initial state for temporal analysis.")

    for k, v in arguments.items():
        # Handle boolean flags correctly
        if isinstance(v.get('default'), bool):
             parser.add_argument(k, action=argparse.BooleanOptionalAction, default=v.get('default'), help=v.get('metadata', {}).get('help', ''))
        else:
             parser.add_argument(k, **v)

    args = parser.parse_args()
    conf = create_experiment_from_args(args, UDRLExperiment)

    # --- 2. Find Experiment Files ---
    exp_name = dataclass_non_defaults_to_string(conf)
    #base_path = Path("data") / conf.env_name / exp_name / str(conf.seed) # Use config to find path

    # *** Example: Override base_path if needed for testing ***
    base_path = Path("data") / "AhnChemoEnv-discrete-setting1/max_episode15/42"
  
    policy_path = base_path / "policy.pkl"
    background_data_path = base_path / "testing_data.csv"
    plot_dir = base_path / "xai_plots_fixed_state" if args.use_fixed_state else base_path / "xai_plots" # Separate plot dir

    # if not policy_path.exists() or not background_data_path.exists():
    #     print(f"Error: Could not find required files in {base_path}")
    #     print("Please run the training script (dtrimpl.py) first.")
    #     exit()

    print(f"Analyzing experiment: {base_path}")
    if args.use_fixed_state:
        print("--- Running FIXED STATE analysis ---")

    # --- 3. Setup ---
    env = gym.make(conf.env_name, n_act=conf.env_n_act) # Use config

    num_state_features = env.observation_space.shape[0]
    num_actions = env.action_space.n

    feature_names = [f"state_{i}" for i in range(num_state_features)] + [r"$d_r$", r"$d_h$"]
    class_names = [f"action_{i}" for i in range(num_actions)]

    print(f"Found {num_state_features} state features and {num_actions} actions.")

    # Load policy and agent
    if conf.estimator_name == "neural":
        policy = NeuralPolicy.load(str(policy_path))
        predict_fn = lambda x: predict_fn_nn(x, policy.estimator, num_state_features)
    else:
        policy = SklearnPolicy.load(str(policy_path))
        predict_fn = lambda x: predict_fn_sklearn(x, policy.estimator, num_state_features)

    agent = UpsideDownAgent(conf=apply(AgentHyper, asdict(conf)), policy=policy, env=env)

    # --- 4. Prepare Background & Fixed State ---
    print("Preparing background data...")
    data = pd.read_csv(background_data_path)
    state_cols = [f"state_{i}" for i in range(num_state_features)]
    command_cols = ["desired_return", "desired_horizon"]

    if not all(c in data.columns for c in state_cols):
        print("Error: Mismatch between env state features and CSV file columns.")
        exit()

    X_state = data[state_cols].values.astype("float32")
    X_command = data[command_cols].values.astype("float32")
    X_combined = np.hstack([X_state, X_command])
    background_data = shap.sample(X_combined, 100)

    # --- NEW: Get fixed state if requested ---
    fixed_initial_state = None
    if args.use_fixed_state:
        # Take the state from the very first row of the testing data
        fixed_initial_state = X_state[0, :].copy()
        print(f"Using fixed initial state: {fixed_initial_state}")
    # --- END NEW ---

    # --- 5. Initialize Explainers ---
    print("Initializing SHAP and LIME explainers...")
    shap_explainer = shap.KernelExplainer(predict_fn, background_data)
    lime_explainer = lime_tabular.LimeTabularExplainer(
        background_data, mode='classification', feature_names=feature_names,
        class_names=class_names, discretize_continuous=True
    )

    # --- 6. Determine Test Goals ---
    # (Using simplified fixed goal logic here for clarity, your dynamic logic is also fine)
    initial_desired_return = conf.final_desired_return
    initial_desired_horizon = conf.final_desired_horizon
    print(f"Using test goal: dr={initial_desired_return}, dh={initial_desired_horizon}")

    # --- 7. Run Simulations ---
    all_episodes_shap, all_episodes_lime = [], []

    print(f"Running {args.num_simulations} simulations for XAI...")
    for sim_num in trange(args.num_simulations, desc="Simulations"):
        # --- MODIFIED RESET LOGIC ---
        state_raw, _ = env.reset(seed=sim_num) # Standard reset
        if args.use_fixed_state and fixed_initial_state is not None:
            # Manually override the state AFTER reset
            try:
                env.unwrapped.state = fixed_initial_state.copy()
                state_raw = env.unwrapped.state # Use the overridden state
            except AttributeError:
                 print("Warning: Could not set env.unwrapped.state. Environment might not support manual state setting.")
                 # Continue with the state returned by reset()
        state = state_raw # Use the potentially overridden state
        # --- END MODIFIED RESET LOGIC ---

        terminated, truncated = False, False
        episode_shap_values, episode_lime_values = [], []
        desired_return = float(initial_desired_return)
        desired_horizon = float(initial_desired_horizon)

        while not (terminated or truncated):
            state_expanded = np.expand_dims(state, axis=0)
            scaled_dr = desired_return * conf.return_scale
            scaled_dh = desired_horizon * conf.horizon_scale
            command_scaled = np.array([[scaled_dr, scaled_dh]])
            ext_state_scaled = np.concatenate((state_expanded, command_scaled), axis=1)

            # --- Get action ONCE ---
            action_probs = predict_fn(ext_state_scaled)[0]
            action = np.argmax(action_probs)

            # --- Calculate explanations ---
            shap_values_instance = shap_explainer.shap_values(ext_state_scaled)
            if isinstance(shap_values_instance, list):
                shap_value_to_use = np.squeeze(shap_values_instance[action])
            else: # Should not happen with softmax fix, but keep for robustness
                shap_value_to_use = np.squeeze(shap_values_instance)
            episode_shap_values.append(shap_value_to_use)

            explanation = lime_explainer.explain_instance(
                ext_state_scaled[0], predict_fn,
                num_features=len(feature_names), labels=(action,)
            )
            lime_weights = np.zeros(len(feature_names))
            lime_map = dict(explanation.as_map().get(action, []))
            for i in range(len(feature_names)):
                lime_weights[i] = lime_map.get(i, 0.0)
            episode_lime_values.append(lime_weights)

            # --- Step the environment ---
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state # Use the raw next_state
            desired_return -= reward
            desired_horizon = max(0, desired_horizon - 1)

        all_episodes_shap.append(episode_shap_values)
        all_episodes_lime.append(episode_lime_values)

    env.close()

    # --- 8. Summarize and Plot ---
    plot_suffix = "_fixed_state" if args.use_fixed_state else ""
    if all_episodes_shap:
        print("\nSummarizing and plotting SHAP explanations...")
        summarized_shap = summarize_explanations(all_episodes_shap)
        plot_explanations_and_create_gifs(summarized_shap, plot_dir / "shap", f"SHAP{plot_suffix}", feature_names)

    if all_episodes_lime:
        print("\nSummarizing and plotting LIME explanations...")
        summarized_lime = summarize_explanations(all_episodes_lime)
        plot_explanations_and_create_gifs(summarized_lime, plot_dir / "lime", f"LIME{plot_suffix}", feature_names)

    print(f"\nScript finished. Plots saved to: {plot_dir}")

