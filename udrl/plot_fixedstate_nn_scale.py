
import gymnasium as gym
import torch
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
from udrl.policies import NeuralPolicy
from udrl.agent import UpsideDownAgent, AgentHyper
from pathlib import Path
from itertools import zip_longest
from tqdm import tqdm, trange
import imageio
from lime import lime_tabular

# ==================================
# 1. Explainer and Plotting Functions
# ==================================

def predict_fn(inputs, model, return_scale, horizon_scale):
    """
    This prediction function contains the SECOND layer of scaling.
    It receives already-scaled data from the explainer and incorrectly scales it again.
    """
    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, axis=0)

    num_state_features = 4
    state = torch.tensor(inputs[:, :num_state_features], dtype=torch.float32)
    command = torch.tensor(inputs[:, num_state_features:], dtype=torch.float32)

    # --- ⚠️ DOUBLE SCALING STEP 2: The predict_fn scales the data AGAIN ---
    command[:, 0] *= return_scale
    command[:, 1] *= horizon_scale
    # ----------------------------------------------------------------------

    model.eval()
    with torch.no_grad():
        probs = model(state, command).numpy()
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

def plot_explanations_and_create_gifs(explanations, plot_dir, title_prefix):
    """Plots the average explanations and creates a GIF from the plots."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    feature_names = ["x", "ẋ", "θ", "θ*", r"$d_t^r$", r"$d_t^h$"]
    image_filenames = []

    for idx, feat_data in tqdm(
        enumerate(explanations), desc=f"Plotting {title_prefix}", total=len(explanations)
    ):
        if not feat_data:
            continue

        x_pos = np.arange(len(feat_data))
        plt.figure(figsize=(10, 6))
        # Add a warning to the plot title
        plt.title(f"{title_prefix} Explanations at Time Step {idx} (Scaled)")

        means = [x[0] for x in feat_data]
        stds = [x[1] for x in feat_data]

        plt.bar(x_pos, means, yerr=stds, capsize=5)
        plt.xticks(x_pos, feature_names)
        plt.xlabel("Features")
        plt.ylabel(f"Mean Absolute {title_prefix} Value")
        plt.tight_layout()
        
        filepath = plot_dir / f"{title_prefix.lower()}_timestep_{idx}.png"
        plt.savefig(filepath)
        plt.close()
        image_filenames.append(filepath)

    # --- Create GIF ---
    gif_path = plot_dir.parent / f"{title_prefix.lower()}_doublescaled_animation.gif"
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
    # --- 1. Configuration ---
    CONF = {
        "env_name": "CartPole-v1",
        "num_simulations": 100,
        "initial_desired_return": 200,
        "initial_desired_horizon": 200,
        "return_scale": 0.02,
        "horizon_scale": 0.02,
        "fixed_initial_state": np.array([0.1, 0.1, 0.1, 0.1]),
        "policy_path": Path(r"data/CartPole-v0/estimator_nameneural_max_episode502_batch_size256_warm_up265/42/policy"),
        "background_data_path": Path("data/CartPole-v0/estimator_nameneural_max_episode502_batch_size256_warm_up265/42/testing_data.csv"),
        "plot_dir": Path("./nn_doublescaled_plots"),
    }
    
    # --- 2. Setup ---
    env = gym.make(CONF["env_name"])
    policy = NeuralPolicy.load(str(CONF["policy_path"]))
    hyper_params = AgentHyper(
        env_name=CONF["env_name"], warm_up=0,
        return_scale=CONF["return_scale"], horizon_scale=CONF["horizon_scale"]
    )
    agent = UpsideDownAgent(hyper_params, policy)

    # --- 3. Prepare Background Data ---
    print("Preparing background data for SHAP and LIME...")
    data = pd.read_csv(CONF["background_data_path"])
    state_cols = data.columns[data.columns.str.startswith("state")].tolist()
    command_cols = data.columns[data.columns.str.startswith("desired")].tolist()
    
    X_state = data[state_cols].values.astype("float32")
    X_command = data[command_cols].values.astype("float32")

    # --- ⚠️ DOUBLE SCALING STEP 1A: Scale the background data commands BEFORE creating the explainer ---
    print("Applying FIRST layer of scaling to background data...")
    X_command[:, 0] *= CONF["return_scale"]
    X_command[:, 1] *= CONF["horizon_scale"]
    # -------------------------------------------------------------------------------------------------

    X_combined = np.hstack([X_state, X_command])
    background_data = shap.sample(X_combined, 100)

    # --- 4. Initialize Explainers ---
    feature_names = ["x", "ẋ", "θ", "θ*", r"$d_t^r$", r"$d_t^h$"]
    shap_explainer = shap.KernelExplainer(
        lambda x: predict_fn(x, policy.estimator, CONF["return_scale"], CONF["horizon_scale"]),
        background_data
    )
    lime_explainer = lime_tabular.LimeTabularExplainer(
        background_data, mode='classification', feature_names=feature_names,
        class_names=['left', 'right'], discretize_continuous=True
    )

    # --- 5. Run Simulations ---
    all_episodes_shap, all_episodes_lime = [], []

    print(f"Running {CONF['num_simulations']} simulations...")
    for sim_num in trange(CONF["num_simulations"], desc="Simulations"):
        state, _ = env.reset(seed=sim_num)
        env.unwrapped.state = CONF["fixed_initial_state"].copy()
        state = env.unwrapped.state

        terminated, truncated = False, False
        episode_shap_values, episode_lime_values = [], []
        
        desired_return = float(CONF["initial_desired_return"])
        desired_horizon = float(CONF["initial_desired_horizon"])

        while not (terminated or truncated):
            state_expanded = np.expand_dims(state, axis=0)
            command = np.array([desired_return, desired_horizon])
            command_expanded = np.expand_dims(command, axis=0)
            ext_state_unscaled = np.concatenate((state_expanded, command_expanded), axis=1)

            # --- ⚠️ DOUBLE SCALING STEP 1B: Scale the instance to be explained BEFORE passing it to the explainer ---
            ext_state_scaled = ext_state_unscaled.copy()
            ext_state_scaled[:, 4] *= agent.conf.return_scale    # Scale desired_return
            ext_state_scaled[:, 5] *= agent.conf.horizon_scale  # Scale desired_horizon
            # ------------------------------------------------------------------------------------------------------

            # --- Calculate explanations using the SCALED data ---
            shap_values_instance = shap_explainer.shap_values(ext_state_scaled)[0]
            episode_shap_values.append(np.squeeze(shap_values_instance))

            explanation = lime_explainer.explain_instance(
                ext_state_scaled[0],
                lambda x: predict_fn(x, policy.estimator, agent.conf.return_scale, agent.conf.horizon_scale),
                num_features=ext_state_scaled.shape[1], labels=(1,)
            )
            lime_weights = np.zeros(ext_state_scaled.shape[1])
            lime_map = dict(explanation.as_map().get(1, []))
            for i in range(ext_state_scaled.shape[1]):
                lime_weights[i] = lime_map.get(i, 0.0)
            episode_lime_values.append(lime_weights)

            # --- Get action (this part MUST use correctly scaled commands to work) ---
            scaled_command = np.array(
                [[desired_return * agent.conf.return_scale, desired_horizon * agent.conf.horizon_scale]],
                dtype=np.float32
            )
            action = agent.policy(state=state_expanded, command=scaled_command, test=True)

            # --- Step the environment and update state ---
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            desired_return -= reward
            desired_horizon = max(0, desired_horizon - 1)

        all_episodes_shap.append(episode_shap_values)
        all_episodes_lime.append(episode_lime_values)

    env.close()

    # --- 6. Summarize and Plot ---
    if all_episodes_shap:
        print("\nSummarizing and plotting SHAP explanations (from scaled data)...")
        summarized_shap = summarize_explanations(all_episodes_shap)
        plot_explanations_and_create_gifs(summarized_shap, CONF["plot_dir"] / "shap", "SHAP")

    if all_episodes_lime:
        print("\nSummarizing and plotting LIME explanations (from scaled data)...")
        summarized_lime = summarize_explanations(all_episodes_lime)
        plot_explanations_and_create_gifs(summarized_lime, CONF["plot_dir"] / "lime", "LIME")

    print("\nScript finished.")