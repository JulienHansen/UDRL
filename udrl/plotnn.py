import torch
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
from udrl.policies import NeuralPolicy
from udrl.agent import UpsideDownAgent, AgentHyper
from pathlib import Path
from itertools import zip_longest, tee
from tqdm import tqdm, trange
import imageio
from lime import lime_tabular



# ==================================
# 1. Helper Functions
# ==================================

# Keep your existing predict_fn, summarize_episodes_shap, and example_plot_shap functions.
# They are fine as they are. You will need a new plotting function for LIME.
def calculate_ep_lime_importance(episode, agent, lime_explainer, desired_return, desired_horizon):
    """
    Calculates LIME explanations for each time step in a single episode.
    """
    ep_lime_values = []

    # Define the scales here
    return_scale = 0.02
    horizon_scale = 0.02

    for state, _, reward in zip(*episode.values()):
        # Create the full input for this time step AND SCALE IT IMMEDIATELY
        command = np.array(
            [
                desired_return * return_scale,   # <-- FIX: Scale here
                desired_horizon * horizon_scale, # <-- FIX: Scale here
            ]
        )
        ext_state = np.concatenate((state, command.reshape(1, -1)), axis=1)

        # The LIME explanation call now receives correctly scaled data
        explanation = lime_explainer.explain_instance(
            ext_state[0],
            lambda x: predict_fn(x, agent.policy.estimator),
            num_features=ext_state.shape[1]
        )
        
        # This part remains the same
        lime_values = np.zeros(ext_state.shape[1])
        for feature, weight in explanation.as_map()[1]: # [1] for the positive class
            lime_values[feature] = weight
        
        ep_lime_values.append(lime_values)
        
        # Update UNscaled variables for the next step's logic
        desired_return -= reward
        desired_horizon = max(desired_horizon - 1, 1)

    return ep_lime_values

def summarize_episodes_lime(episodes_lime, summarize_funs: list = [np.mean, np.std]):
    """
    Averages LIME explanations across multiple episodes for each time step.
    This is very similar to summarize_episodes_shap.
    """
    transposed_data = list(zip_longest(*episodes_lime, fillvalue=None))
    summary = []
    for time_step_data in transposed_data:
        valid_data = [d for d in time_step_data if d is not None]
        if not valid_data:
            summary.append([])
            continue
        features_at_timestep = list(zip(*valid_data))
        feature_summary = []
        for feature_data in features_at_timestep:
            feature_summary.append([fun(feature_data) for fun in summarize_funs])
        summary.append(feature_summary)
    return summary


def example_plot_lime(lime_importances, plot_dir):
    """
    Plots the average LIME importance for each feature over time.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)
    feature_names = ["x", "ẋ", "θ", "θ*", r"$d_t^r$", r"$d_t^h$"]
    
    for idx, lime_feat in tqdm(
        enumerate(lime_importances),
        desc="Plotting LIME",
        total=len(lime_importances),
    ):
        if not lime_feat:
            continue
        x_pos = np.arange(len(lime_feat))
        plt.figure()
        plt.title(f"NN LIME Explanations at Time Step {idx}")
        
        # We plot the absolute LIME values
        means = [np.abs(x[0]) for x in lime_feat]
        stds = [x[1] for x in lime_feat]
        
        plt.bar(x_pos, means, yerr=stds)
        plt.xticks(x_pos, feature_names)
        plt.xlabel("Features")
        plt.ylabel("Average Absolute LIME Weight")
        plt.savefig(plot_dir / f"lime_timestep_{idx}.png")
        plt.close()



def calculate_ep_shap_importance(episode, agent, explainer, desired_return, desired_horizon):
    """
    Calculates SHAP values for each time step in a single episode.
    """
    ep_shap_values = []
    
    # Define the scales here, ideally passed in or from a config
    return_scale = 0.02 
    horizon_scale = 0.02
    
    for state, _, reward in zip(*episode.values()):
        # Create the full input for this time step AND SCALE IT IMMEDIATELY
        command = np.array(
            [
                desired_return * return_scale,   # <-- FIX: Scale here
                desired_horizon * horizon_scale, # <-- FIX: Scale here
            ]
        )
        ext_state = np.concatenate((state, command.reshape(1, -1)), axis=1)

        # This part remains the same
        shap_values_instance = explainer.shap_values(ext_state)[0]
        ep_shap_values.append(shap_values_instance)
        
        # Update UNscaled variables for the next step's logic
        desired_return -= reward
        desired_horizon = max(desired_horizon - 1, 1)

    return ep_shap_values



def summarize_episodes_shap(episodes_shap, summarize_funs: list = [np.mean, np.std]):
    """
    Averages SHAP values across multiple episodes for each time step.
    """
    # The structure will be: time_step -> [feature1_list, feature2_list, ...]
    transposed_data = list(zip_longest(*episodes_shap, fillvalue=None))

    # The final summarized data will be: time_step -> [[mean, std], [mean, std], ...]
    summary = []
    for time_step_data in transposed_data:
        # Filter out None values for episodes that ended early
        valid_data = [d for d in time_step_data if d is not None]
        
        if not valid_data:
            summary.append([]) # Append an empty list if no data for this time step
            continue

        # Transpose to group by feature, e.g., [[val1, val2], [val1, val2]] -> [[val1, val1], [val2, val2]]
        features_at_timestep = list(zip(*[np.squeeze(arr) for arr in valid_data]))
        
        feature_summary = []
        for feature_data in features_at_timestep:
            feature_summary.append([fun(feature_data) for fun in summarize_funs])
        
        summary.append(feature_summary)

    return summary


def example_plot_shap(shap_importances, plot_dir):
    """
    Plots the average SHAP importance for each feature over time.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Feature labels
    feature_names = ["x", "ẋ", "θ", "θ*", r"$d_t^r$", r"$d_t^h$"] # Adjust for Acrobot if needed
    
    for idx, shap_feat in tqdm(
        enumerate(shap_importances),
        desc="Plotting SHAP",
        total=len(shap_importances),
    ):
        if not shap_feat:  # Skip if no data for this time step
            continue

        x_pos = np.arange(len(shap_feat))
        plt.figure()
        plt.title(f"NN SHAP Values at Time Step {idx}")
        
        # We plot the absolute SHAP values for importance
        means = [np.abs(x[0]) for x in shap_feat]
        stds = [x[1] for x in shap_feat]
        
        plt.bar(x_pos, means, yerr=stds)
        plt.xticks(x_pos, feature_names)
        plt.xlabel("Features")
        plt.ylabel("Average Absolute SHAP Value")
        plt.savefig(plot_dir / f"shap_timestep_{idx}.png")
        plt.close()


# ==================================
# 2. Main Script
# ==================================


# --- Helper function ---
def predict_fn(inputs, model):
    num_state_features = 4  # CartPole
    state = torch.tensor(inputs[:, :num_state_features], dtype=torch.float32)
    command = torch.tensor(inputs[:, num_state_features:], dtype=torch.float32)

    # Apply the same scaling used in agent.collect_episode()
    # You would need to make sure the agent's scales are accessible
    return_scale = 0.02
    horizon_scale = 0.02
    command[:, 0] *= return_scale
    command[:, 1] *= horizon_scale

    model.eval()
    with torch.no_grad():
        probs = model(state, command).numpy()
    return probs


def create_gif_from_plots(image_filenames, output_filename, duration=0.5):
    """
    Creates a GIF from a list of image files.
    
    Args:
        image_filenames (list): A list of strings, where each string is the
                                path to an image file.
        output_filename (str): The path to save the output GIF file.
        duration (float): The duration in seconds to display each image.
    """
    try:
        images = [imageio.imread(f) for f in image_filenames]
        imageio.mimsave(output_filename, images, duration=duration)
        print(f"GIF saved to {output_filename}")
    except FileNotFoundError as e:
        print(f"Error creating GIF: {e}. One or more image files were not found.")
    except Exception as e:
        print(f"An unexpected error occurred while creating the GIF: {e}")


if __name__ == "__main__":
    # --- Configuration ---
    base_path = Path("data")
    env = "CartPole-v0"
    estimator_name = "neural"
    seed = "42"
    conf_name = "estimator_nameneural"
    desired_return = 200
    desired_horizon = 200
    horizon_scale = 0.02
    return_scale = 0.02
    redundancy = 100

    path = base_path / env / conf_name / seed
    plot_dir = base_path / "example_shap_plot"
    
    # --- Load Policy and Agent ---
    policy = NeuralPolicy.load(str(path / "policy"))
    agent = UpsideDownAgent(
        AgentHyper(env, warm_up=50),
        policy
    )

    # --- Load Data for SHAP Background Set ---
    data_path = path / "testing_data.csv"
    data = pd.read_csv(data_path)
    
    state_cols = data.columns[data.columns.str.startswith("state")].tolist()
    command_cols = data.columns[data.columns.str.startswith("desired")].tolist()
    X_state = data[state_cols].values.astype("float32")
    X_command = data[command_cols].values.astype("float32")

     # --- THIS IS THE FIX ---
    # Scale the command columns in your dataset before doing anything else.
    X_command[:, 0] *= return_scale
    X_command[:, 1] *= horizon_scale
    
    X_combined = np.hstack([X_state, X_command])


    # Use a small, representative sample for the background set.
    # Pass a NumPy array directly.
    background_data_shap = shap.sample(X_combined, 200) 
    
    # --- Initialize SHAP Explainer ---
    # Pass the function and background data.
    explainer = shap.KernelExplainer(lambda x: predict_fn(x, policy.estimator), background_data_shap)

    feature_names = ["x", "ẋ", "θ", "θ*", r"$d_t^r$", r"$d_t^h$"]

    # LIME Explainer
    # Use the same background data for consistency
    explainer_lime = lime_tabular.LimeTabularExplainer(
        background_data_shap, # This serves as the training data for LIME
        mode='classification', 
        feature_names=feature_names,
        class_names=['left', 'right'], # Adjust for your problem's classes
        discretize_continuous=True
    )


    # --- Collect Episodes and Calculate Explanations ---
    for _ in trange(redundancy, desc="Collect Data"):
        agent.collect_episode(desired_return, desired_horizon, test=True)
    
    episodes = [
        {k: v for k, v in ep.items() if k != "summed_rewards"}
        for ep in agent.memory.buffer
    ]
    
    episodes_shap_values = []
    episodes_lime_values = []
    
    for ep in tqdm(episodes, desc="Calculate SHAP and LIME"):
        # Calculate SHAP values
        ep_shap = calculate_ep_shap_importance(
            ep, agent, explainer, desired_return, desired_horizon
        )
        episodes_shap_values.append(ep_shap)
        
        # Calculate LIME explanations
        ep_lime = calculate_ep_lime_importance(
            ep, agent, explainer_lime, desired_return, desired_horizon
        )
        episodes_lime_values.append(ep_lime)

    # --- Summarize and Plot SHAP ---
    shap_importances = summarize_episodes_shap(episodes_shap_values)
    example_plot_shap(shap_importances, plot_dir / "shap")

    # --- Summarize and Plot LIME ---
    lime_importances = summarize_episodes_lime(episodes_lime_values)
    example_plot_lime(lime_importances, plot_dir / "lime")
    
    
    # --- Create GIF ---
    image_filenames = [
        str(plot_dir / "shap" / f"shap_timestep_{idx}.png")
        for idx in range(len(shap_importances))
        if shap_importances[idx] # ensure data exists for the time step
    ]
        
    # Check if any images were generated
    if image_filenames:
        imageio.mimsave(str(plot_dir / "shap_animation.gif"), [imageio.imread(f) for f in image_filenames], duration=0.5)
        print("GIF created successfully!")
    else:
        print("No images were generated for the GIF.")

    # --- Create GIF for LIME ---
    image_filenames_lime = [
        str(plot_dir / "lime" / f"lime_timestep_{idx}.png")
        for idx in range(len(lime_importances))
        if lime_importances[idx]  # ensure data exists for the time step
    ]

    if image_filenames_lime:
        create_gif_from_plots(
            image_filenames_lime,
            output_filename=str(plot_dir / "lime_animation.gif"),
            duration=0.5
        )
        print("LIME GIF created successfully!")
    else:
        print("No images were generated for the LIME GIF.")