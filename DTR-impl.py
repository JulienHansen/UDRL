import pandas as pd
import time
import json
import torch
import random as rnd
import numpy as np
import argparse
from tqdm import trange
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

import gymnasium as gym
import DTRGym  # Necessary import to register the environments

# Import from your UDRL project library
from udrl.agent import UpsideDownAgent, AgentHyper
from udrl.policies import SklearnPolicy, NeuralPolicy
from udrl.cli import (
    with_meta,
    create_argparse_dict,
    create_experiment_from_args,
    dataclass_non_defaults_to_string,
    apply,
)

# Configuration dataclass with defaults tuned for DTR
@dataclass
class UDRLExperiment:
    env_name: str = with_meta("SimGlucoseEnv_discrete_setting1", "Name of the DTR-Bench environment")
    env_n_act: int = with_meta(5, "Number of actions for the environment")
    estimator_name: str = with_meta("neural", "Policy model: 'neural' or scikit-learn classifier")
    epsilon: float = with_meta(0.2, "Exploration rate for epsilon-greedy action selection")
    seed: int = with_meta(42, "Random seed")
    max_episode: int = with_meta(20, "Maximum number of training episodes")
    collect_iter: int = with_meta(10, "Episodes to collect between training updates")
    train_per_iter: int = with_meta(10, "Training iterations per collection step")
    batch_size: int = with_meta(512, "Batch size for sampling from the replay buffer")
    warm_up: int = with_meta(50, "Number of initial random *episodes* to populate the buffer")
    memory_size: int = with_meta(50000, "Maximum size of the replay buffer")
    last_few: int = with_meta(100, "Number of recent episodes for exploratory command sampling")
    
    # --- ADD THIS ---
    determine_test_commands: bool = with_meta(True, "Dynamically determine test dr/dh from last episodes, per paper ")
    
    # --- MODIFY THESE ---
    final_desired_return: int = with_meta(-500, "Target clinical score for testing (FALLBACK if determine_test_commands=False)")
    final_desired_horizon: int = with_meta(90, "Target treatment duration for testing (FALLBACK if determine_test_commands=False)")
    horizon_scale: float = with_meta(0.01, "Scaling factor for desired horizon")
    return_scale: float = with_meta(0.01, "Scaling factor for desired return")
    testing_period: int = with_meta(50, "How often to run testing episodes")
    final_testing: bool = with_meta(True, "Perform final testing after training")
    final_testing_sample: int = with_meta(100, "Number of episodes for final evaluation")
    save_policy: bool = with_meta(True, "Save the final trained policy")
    save_learning_infos: bool = with_meta(True, "Save learning logs and reward curves")
    save_test_data: bool = with_meta(True, "Save the full trajectories from final testing")

def dump_dict(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def run_experiment(conf: UDRLExperiment):
    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)
    rnd.seed(conf.seed)

    env = gym.make(conf.env_name, n_act=conf.env_n_act)
    
    if conf.estimator_name == "neural":
        policy = NeuralPolicy(env.observation_space.shape[0], action_size=env.action_space.n)
    else:
        policy = SklearnPolicy(estimator_name=conf.estimator_name, action_size=env.action_space.n, epsilon=conf.epsilon)
        
    agent = UpsideDownAgent(conf=apply(AgentHyper, asdict(conf)), policy=policy, env=env)
    
    print(f"Warming up replay buffer with {conf.warm_up} random episodes...")
    for _ in trange(conf.warm_up, desc="Warm-up Phase"):
        agent.collect_episode(*agent.sample_exploratory_commands())

    epi_bar = trange(conf.max_episode)
    returns, test_returns, infos = [], [], []
    test_reward_mean, test_reward_std = 0, 0

    for e in epi_bar:
        # Collect new data
        episodic_rewards = []
        for _ in range(conf.collect_iter):
            r, _, _ = agent.collect_episode(*agent.sample_exploratory_commands()) # <--- MODIFY THIS
            episodic_rewards.append(r) # <--- ADD THIS
        
        # Store the mean and std of the rewards from this training iteration
        if episodic_rewards:
            returns.append((np.mean(episodic_rewards), np.std(episodic_rewards))) # <--- ADD THIS
        
        # Train on the buffer
        metric = []
        if len(agent.memory.buffer) >= conf.batch_size:
            for _ in range(conf.train_per_iter):
                info = agent.train()
                metric.append(info["metric"])
                infos.append(info)
        
        # Periodically test the agent
        if e > 0 and e % conf.testing_period == 0:
            test_rewards = [
                agent.collect_episode(conf.final_desired_return, conf.final_desired_horizon, test=True, store_episode=False)[0]
                for _ in range(conf.final_testing_sample)
            ]
            test_reward_mean, test_reward_std = np.mean(test_rewards), np.std(test_rewards)
            test_returns.append((test_reward_mean, test_reward_std))

        epi_bar.set_postfix({"Test Reward": f"{test_reward_mean:.2f} ± {test_reward_std:.2f}", "Metric": np.mean(metric or [0])})


    if conf.final_testing:
        if conf.determine_test_commands:
            print(f"\nDetermining final test commands from last {conf.last_few} episodes...")
            
            # Get the last 'conf.last_few' episodes from the replay buffer
            num_episodes_to_sample = min(len(agent.memory.buffer), conf.last_few)
            
            if num_episodes_to_sample > 0:
                last_episodes = agent.memory.buffer[-num_episodes_to_sample:]
                
                # Calculate the initial dr (total return) and dh (total horizon) for each episode
                all_drs = [sum(ep["rewards"]) for ep in last_episodes]
                all_dhs = [len(ep["states"]) for ep in last_episodes]
                
                # Use the mean to interpret the "most common" value 
                final_test_dr = np.mean(all_drs)
                final_test_dh = int(np.mean(all_dhs)) # Horizon should be an integer
                
                print(f"Using dynamically determined test commands: dr={final_test_dr:.2f}, dh={final_test_dh}")
            else:
                print(f"Warning: Replay buffer is empty. Falling back to config defaults.")
                final_test_dr = conf.final_desired_return
                final_test_dh = conf.final_desired_horizon
        else:
            print(f"\nUsing fixed test commands from config: dr={conf.final_desired_return}, dh={conf.final_desired_horizon}")
            final_test_dr = conf.final_desired_return
            final_test_dh = conf.final_desired_horizon
    else:
        # Set defaults if final_testing is False, though they won't be used
        final_test_dr = conf.final_desired_return
        final_test_dh = conf.final_desired_horizon
    # --- END OF NEW BLOCK ---

    # --- DATA SAVING LOGIC ---
    exp_name = dataclass_non_defaults_to_string(conf)
    base_path = Path("data") / conf.env_name / exp_name / str(conf.seed)
    base_path.mkdir(parents=True, exist_ok=True)
    print(f"\nTraining complete. Saving all artifacts to: {base_path}")

    if conf.save_policy:
        agent.policy.save(str(base_path / "policy.pkl"))

    # (The full data saving logic from your original main.py is included below)
    # ... (This includes saving training_data.csv, testing_data.csv, and all the .npy files)
    final_res = {}
    if conf.final_testing:
        print("Start Final Testing...")
        final_rewards = []
        all_test_features = []
        all_test_targets = []

        for sim_num in trange(conf.final_testing_sample, desc="Final Testing"):
            state, _ = env.reset()
            done, truncated = False, False
            cum_rew = 0
            steps = 0

            while not (done or truncated):
                # USE THE NEW DYNAMIC VALUES (final_test_dr, final_test_dh)
                desired_return = final_test_dr - cum_rew
                desired_horizon = final_test_dh - steps

                state_expanded = np.expand_dims(state, axis=0)
                command = np.array([
                    desired_return * conf.return_scale,
                    desired_horizon * conf.horizon_scale,
                ])
                command_expanded = np.expand_dims(command, axis=0)
                action = agent.policy(state_expanded, command_expanded, test=True)

                # store for DataFrame
                feature_list = list(state)
                feature_list.extend([
                    desired_return * conf.return_scale,
                    desired_horizon * conf.horizon_scale,
                    sim_num,
                    time.time()
                ])
                all_test_features.append(feature_list)
                all_test_targets.append(action)

                next_state, reward, done, truncated, _ = env.step(action)
                state = next_state
                cum_rew += reward
                steps += 1

            final_rewards.append(cum_rew)

        final_res["test_mean"] = np.mean(final_rewards)
        final_res["test_std"] = np.std(final_rewards)
        print(f"Final result: {final_res['test_mean']} ± {final_res['test_std']}")

    if conf.save_test_data and all_test_features:
        feature_names = [f"state_{i}" for i in range(env.observation_space.shape[0])]
        feature_names += ["desired_return", "desired_horizon", "simulation_number", "timestamp"]
        df = pd.DataFrame(all_test_features, columns=feature_names)
        df["action"] = all_test_targets
        df.to_csv(base_path / "testing_data.csv", index=False)
        print("Testing data saved successfully to testing_data.csv.")


    if conf.save_learning_infos:
        print("Saving training data for interpretability analysis...")

        # Check if replay buffer has data
        if not agent.memory.buffer:
            print("Warning: Replay buffer is empty. No training data to save.")
        else:
            sample_size = min(len(agent.memory.buffer), 1000)
            random_episodes = agent.memory.get_random_samples(sample_size)

            all_features = []
            all_targets = []

            for sim_num, episode in enumerate(random_episodes):
                T = len(episode["states"])
                for t1 in range(T - 1):
                    t2 = T
                    state = episode["states"][t1]
                    desired_return = sum(episode["rewards"][t1:t2])
                    desired_horizon = t2 - t1
                    action = episode["actions"][t1]

                    feature_list = list(state[0]) if isinstance(state, np.ndarray) and state.ndim == 2 else list(state)
                    feature_list.extend([
                        desired_return * conf.return_scale,
                        desired_horizon * conf.horizon_scale,
                        sim_num,
                        time.time()
                    ])
                    all_features.append(feature_list)
                    all_targets.append(action)

            if all_features:
                feature_names = [f"state_{i}" for i in range(env.observation_space.shape[0])]
                feature_names += ["desired_return", "desired_horizon", "simulation_number", "timestamp"]
                train_df = pd.DataFrame(all_features, columns=feature_names)
                train_df["action"] = all_targets
                train_df.to_csv(base_path / "training_data.csv", index=False)
                print("Training data saved successfully to training_data.csv.")
            else:
                print("Warning: No valid samples found; training CSV not created.")

    np.save(str(base_path / "train_rewards.npy"), returns)
    np.save(str(base_path / "test_rewards.npy"), test_returns)
    dump_dict(infos, str(base_path / "learning_infos.json"))
    print("Learning artifacts and training data sample saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs a UDRL experiment on a DTR-Bench environment.")
    arguments = create_argparse_dict(UDRLExperiment)
    for k, v in arguments.items():
        # NOTE: If booleans cause issues, you may need to add this:
        if isinstance(v.get('default'), bool):
            parser.add_argument(
                k, 
                action=argparse.BooleanOptionalAction, # Makes --no-determine-test-commands work
                default=v.get('default'),
                help=v.get('metadata', {}).get('help', '')
            )
        else:
            parser.add_argument(k, **v)
    
    args = parser.parse_args()
    conf = create_experiment_from_args(args, UDRLExperiment)
    run_experiment(conf)