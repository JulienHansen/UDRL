import gymnasium as gym
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from itertools import zip_longest, tee
import pickle
import imageio
from udrl.policies import SklearnPolicy
from udrl.agent import UpsideDownAgent, AgentHyper


# ==================================
# Feature Importance Functions
# ==================================

def calculate_local_permutation_importance(agent, ext_state):
    """Calculates feature importance by perturbing one feature at a time."""
    base_prediction = agent.policy.estimator.predict(ext_state)
    local_importances = []
    for i in range(ext_state.shape[1]):
        perturbed_state = ext_state.copy()
        # Perturb by adding small random noise
        perturbed_state[:, i] += np.random.normal(0, 0.1)
        perturbed_prediction = agent.policy.estimator.predict(perturbed_state)
        importance = np.sum(base_prediction != perturbed_prediction)
        local_importances.append(importance)
    return local_importances

def calculate_ep_feat_importance(episode_states, episode_commands, agent):
    """
    Calculates both MDI and Permutation importance for each step of an episode.
    """
    ep_mdi_features = []
    ep_perm_features = []
    
    for state, command in zip(episode_states, episode_commands):
        ext_state = np.concatenate((state, command), axis=1)
        
        # --- Calculate MDI Importance ---
        mdi_feature_importances = {}
        if hasattr(agent.policy.estimator, 'estimators_'):
            for t in agent.policy.estimator.estimators_:
                branch = np.array(t.decision_path(ext_state).todense(), dtype=bool)
                if not any(branch[0]): continue
                
                imp = t.tree_.impurity[branch[0]]
                features_on_path = t.tree_.feature[branch[0]][:-1]
                
                for f, i in zip(features_on_path, imp[:-1] - imp[1:]):
                    mdi_feature_importances.setdefault(f, []).append(i)
        
        summed_mdi = [sum(mdi_feature_importances.get(k, [])) for k in range(ext_state.shape[1])]
        ep_mdi_features.append(summed_mdi)
        
        # --- Calculate Permutation Importance ---
        perm_importances = calculate_local_permutation_importance(agent, ext_state)
        ep_perm_features.append(perm_importances)
        
    return ep_mdi_features, ep_perm_features

def summarize_episodes_feat(episodes_feat, summarize_funs=[np.mean, np.std]):
    """
    Averages feature importance values across multiple episodes for each time step.
    """
    transposed_data = list(zip_longest(*episodes_feat, fillvalue=None))
    summary = []
    for time_step_data in transposed_data:
        valid_data = [d for d in time_step_data if d is not None and len(d) > 0]
        if not valid_data:
            summary.append([])
            continue
        
        features_at_timestep = list(zip(*valid_data))
        feature_summary = []
        for feature_data in features_at_timestep:
            feature_summary.append([fun(feature_data) for fun in summarize_funs])
        summary.append(feature_summary)
    return summary

def plot_and_create_gifs(mdi_importances, perm_importances, plot_dir):
    """
    Generates plots and animated GIFs for both MDI and Permutation importance.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)
    mdi_dir = plot_dir / "mdi_importance"
    perm_dir = plot_dir / "permutation_importance"
    mdi_dir.mkdir(exist_ok=True)
    perm_dir.mkdir(exist_ok=True)
    
    feature_names = ["x", "ẋ", "θ", "θ*", r"$d_t^r$", r"$d_t^h$"]

    # --- Plot MDI Importance ---
    for idx, state_feat in tqdm(enumerate(mdi_importances), desc="Plotting MDI", total=len(mdi_importances)):
        if not state_feat: continue
        means, stds = [x[0] for x in state_feat], [x[1] for x in state_feat]
        x_pos = np.arange(len(means))
        plt.figure(figsize=(10, 6))
        plt.bar(x_pos, means, yerr=stds, capsize=5, color='skyblue', ecolor='gray')
        plt.title(f"MDI at Time Step {idx} (Averaged over 100 runs)")
        plt.ylabel("Mean Decrease in Impurity (MDI)")
        plt.xlabel("Features")
        plt.xticks(x_pos, feature_names)
        plt.tight_layout()
        plt.savefig(mdi_dir / f"mdi_importance_{idx}.png")
        plt.close()

    # --- Plot Permutation Importance ---
    for idx, state_feat in tqdm(enumerate(perm_importances), desc="Plotting Permutation", total=len(perm_importances)):
        if not state_feat: continue
        means, stds = [x[0] for x in state_feat], [x[1] for x in state_feat]
        x_pos = np.arange(len(means))
        plt.figure(figsize=(10, 6))
        plt.bar(x_pos, means, yerr=stds, capsize=5, color='lightgreen', ecolor='gray')
        plt.title(f"Permutation Importance at Time Step {idx} (Averaged over 100 runs)")
        plt.ylabel("Local Permutation Importance")
        plt.xlabel("Features")
        plt.xticks(x_pos, feature_names)
        plt.tight_layout()
        plt.savefig(perm_dir / f"perm_importance_{idx}.png")
        plt.close()

    print(f"Plots saved to {mdi_dir} and {perm_dir}")

    # --- Create GIFs ---
    try:
        mdi_images = sorted(mdi_dir.glob('*.png'), key=lambda x: int(x.stem.split('_')[-1]))
        imageio.mimsave(plot_dir / "mdi_animation.gif", [imageio.imread(f) for f in mdi_images], duration=0.5)
        print(f"Successfully created {plot_dir / 'mdi_animation.gif'}")

        perm_images = sorted(perm_dir.glob('*.png'), key=lambda x: int(x.stem.split('_')[-1]))
        imageio.mimsave(plot_dir / "permutation_animation.gif", [imageio.imread(f) for f in perm_images], duration=0.5)
        print(f"Successfully created {plot_dir / 'permutation_animation.gif'}")
    except Exception as e:
        print(f"Could not create GIFs. Error: {e}")

# ==================================
# Main Script Execution
# ==================================

if __name__ == "__main__":
    # --- 1. Configuration ---
    # Replicated the path structure from your original, working code
    base_path = Path("resources")
    env_name = "CartPole-v0" # Using v0 to match original policy path
    conf_name = "RandomForest"
    
    # Construct the full path to the policy directory
    policy_dir_path = base_path / env_name / conf_name / "policy"

    CONF = {
        # Using env_name variable for consistency
        "env_name": "CartPole-v0", # The environment to run
        "num_simulations": 100,
        "initial_desired_return": 200,
        "initial_desired_horizon": 200,
        "return_scale": 0.02,
        "horizon_scale": 0.02,
        "fixed_initial_state": np.array([0.1, 0.1, 0.1, 0.1]),
        # Store the final path object in the config
        "policy_path": policy_dir_path,
        "output_csv_path": Path("./fixed_state_output.csv"),
        "plot_dir": Path("./fixed_state_plots"),
    }

    # --- 2. Setup Environment and Agent ---
    env = gym.make(CONF["env_name"])

    # THE FIX: Convert the Path object to a string before loading
    print(f"Loading policy from: {CONF['policy_path']}")
    policy = SklearnPolicy.load(str(CONF["policy_path"]))
    
    hyper_params = AgentHyper(
        env_name=CONF["env_name"],
        warm_up=0,
        return_scale=CONF["return_scale"],
        horizon_scale=CONF["horizon_scale"],
    )
    agent = UpsideDownAgent(hyper_params, policy)
    
    all_episodes_data = []
    all_episodes_mdi_importances = []
    all_episodes_perm_importances = []

    print(f"Running {CONF['num_simulations']} simulations from fixed state: {CONF['fixed_initial_state']}")
    
    # --- 3. Run Simulations from the Fixed Initial State ---
    for sim_num in trange(CONF["num_simulations"], desc="Simulations"):
        try:
            state, _ = env.reset(seed=sim_num) # Seed for reproducibility
            env.unwrapped.state = CONF["fixed_initial_state"].copy()
            state = env.unwrapped.state
        except Exception:
             state, _ = env.reset() # Fallback for environments that don't allow state setting

        terminated, truncated = False, False
        episode_log, episode_states_for_mdi, episode_commands_for_mdi = [], [], []
        cumulative_reward, time_step = 0, 0

        while not (terminated or truncated):
            state_expanded = np.expand_dims(state, axis=0)
            desired_return = CONF["initial_desired_return"] - cumulative_reward
            desired_horizon = CONF["initial_desired_horizon"] - time_step
            scaled_return = desired_return * CONF["return_scale"]
            scaled_horizon = max(0, desired_horizon) * CONF["horizon_scale"]
            command = np.array([scaled_return, scaled_horizon])
            command_expanded = np.expand_dims(command, axis=0)
            action = agent.policy(state_expanded, command_expanded, test=True)

            log_entry = list(state) + [scaled_return, scaled_horizon, sim_num, action]
            episode_log.append(log_entry)
            episode_states_for_mdi.append(state_expanded)
            episode_commands_for_mdi.append(command_expanded)

            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            cumulative_reward += reward
            time_step += 1
        
        all_episodes_data.extend(episode_log)
        
        mdi_imp, perm_imp = calculate_ep_feat_importance(episode_states_for_mdi, episode_commands_for_mdi, agent)
        all_episodes_mdi_importances.append(mdi_imp)
        all_episodes_perm_importances.append(perm_imp)

    env.close()

    # --- 4. Save Trajectory Data to CSV ---
    if all_episodes_data:
        column_names = ["x", "ẋ", "θ", "θ*", "desired_return", "desired_horizon", "simulation_number", "action"]
        df = pd.DataFrame(all_episodes_data, columns=column_names)
        df.to_csv(CONF["output_csv_path"], index=False)
        print(f"\nTrajectory data for all simulations saved to '{CONF['output_csv_path']}'")

    # --- 5. Analyze and Plot Feature Importances ---
    if all_episodes_mdi_importances:
        print("\nSummarizing feature importances across all simulations...")
        summarized_mdi = summarize_episodes_feat(all_episodes_mdi_importances)
        summarized_perm = summarize_episodes_feat(all_episodes_perm_importances)
        plot_and_create_gifs(summarized_mdi, summarized_perm, CONF["plot_dir"])

