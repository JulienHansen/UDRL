from udrl.policies import SklearnPolicy
from udrl.agent import UpsideDownAgent, AgentHyper
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from itertools import zip_longest, tee
from tqdm import tqdm, trange
import imageio


def calculate_ep_feat_importance(
    episode, agent, desired_return, desired_horizon
):
    ep_features = []
    ep_perm_features = []
    
    # Check if the model is a type that supports decision_path
    #mdi_supported = isinstance(agent.policy.estimator, RandomForestClassifier)
    mdi_supported = True

    for state, _, reward in zip(*episode.values()):
        command = np.array(
            [
                desired_return * agent.conf.return_scale,
                desired_horizon * agent.conf.horizon_scale,
            ]
        )
        command = np.expand_dims(command, axis=0)
        ext_state = np.concatenate((state, command), axis=1)

        # Calculate local MDI importance only if supported
        feature_importances = {}
        if mdi_supported:
            for t in agent.policy.estimator.estimators_:
                branch = np.array(t.decision_path(ext_state).todense(), dtype=bool)
                imp = t.tree_.impurity[branch[0]]
                for f, i in zip(
                    t.tree_.feature[branch[0]][:-1], imp[:-1] - imp[1:]
                ):
                    feature_importances.setdefault(f, []).append(i)
            
            summed_mdi_importances = [
                sum(feature_importances.get(k, []))
                for k in range(ext_state.shape[1])
            ]
            ep_features.append(summed_mdi_importances)
        else:
            # Append empty list if MDI is not supported
            ep_features.append([])

        # Calculate local permutation importance (always runs)
        perm_importances = calculate_local_permutation_importance(agent, ext_state)
        ep_perm_features.append(perm_importances)
        
        # Line 8 Algorithm 2
        desired_return -= reward
        # Line 9 Algorithm 2
        desired_horizon = max(desired_horizon - 1, 1)

    return ep_features, ep_perm_features


def calculate_local_permutation_importance(agent, ext_state):
    base_prediction = agent.policy.estimator.predict(ext_state)
    local_importances = []
    
    # Iterate through each feature
    for i in range(ext_state.shape[1]):
        # Create a perturbed copy
        perturbed_state = ext_state.copy()
        
        # Perturb the feature value by adding random noise
        # This is a local equivalent of shuffling
        perturbed_state[:, i] += np.random.normal(0, 0.1) 
        
        # Get the new prediction
        perturbed_prediction = agent.policy.estimator.predict(perturbed_state)
        
        # The importance is the change in prediction
        importance = np.sum(base_prediction != perturbed_prediction)
        local_importances.append(importance)
        
    return local_importances


def summarize_episodes_feat(
    episodes_feat, summarize_funs: list = [np.mean, np.std]
):
    return [
        [
            [
                fun(list(data))
                for fun, data in zip(
                    summarize_funs,
                    tee(
                        (s for s in state if s is not None),
                        len(summarize_funs),
                    ),
                )
            ]
            for state in zip_longest(*ep)
        ]
        for ep in zip_longest(*episodes_feat, fillvalue=[])
    ]


def calculate_features_importance(
    path: Path,
    env: str,
    desired_return: int,
    desired_horizon: int,
    horizon_scale: float,
    return_scale: float,
    redundancy: int = 100,
):
    policy = SklearnPolicy.load(str(path / "policy"))
    hyper = AgentHyper(
        env,
        warm_up=0,
        horizon_scale=horizon_scale,
        return_scale=return_scale,
    )

    agent = UpsideDownAgent(hyper, policy)

    for _ in trange(redundancy, desc="Collect Data"):
        agent.collect_episode(desired_return, desired_horizon, test=True)

    episodes = [
        {k: v for k, v in ep.items() if k != "summed_rewards"}
        for ep in agent.memory.buffer
    ]

    episodes_feat = []
    episodes_perm_feat = []

    for ep in tqdm(episodes, desc="Calculate importance features"):
        mdi_imp, perm_imp = calculate_ep_feat_importance(
            ep, agent, desired_return, desired_horizon
        )
        episodes_feat.append(mdi_imp)
        episodes_perm_feat.append(perm_imp)

    feature_importances = summarize_episodes_feat(episodes_feat)
    permutation_importances = summarize_episodes_feat(episodes_perm_feat)

    return feature_importances, permutation_importances


def example_plot(feature_importances, permutation_importances, plot_dir):
    # Create the output directories if they don't exist
    mdi_dir = plot_dir / "mdi_importance"
    perm_dir = plot_dir / "permutation_importance"
    mdi_dir.mkdir(parents=True, exist_ok=True)
    perm_dir.mkdir(parents=True, exist_ok=True)
    
    # Feature labels
    feature_names = ["x", "ẋ", "θ", "θ*", r"$d_t^r$", r"$d_t^h$"]

    if any(feature_importances):
        # Plot MDI importance
        for idx, state_feat in tqdm(
            enumerate(feature_importances),
            desc="Plotting MDI",
            total=len(feature_importances),
        ):
            x_pos = np.arange(len(state_feat))
            plt.figure()
            plt.title(f"Cartpole-v0 State {idx}")
            plt.bar(x_pos, [x[0] for x in state_feat], yerr=[x[1] for x in state_feat])
            plt.xticks(x_pos, feature_names)
            plt.savefig(mdi_dir / f"importances_state_{idx}.png")
            plt.close()
    else:
        print("MDI data is empty, skipping MDI plotting.")

    # Plot permutation importance
    for idx, perm_feat in tqdm(
        enumerate(permutation_importances),
        desc="Plotting Permutation",
        total=len(permutation_importances),
    ):
        x_pos = np.arange(len(perm_feat))
        plt.figure()
        plt.title(f"Cartpole-v0 State {idx}")
        plt.bar(x_pos, [x[0] for x in perm_feat], yerr=[x[1] for x in perm_feat])
        plt.xticks(x_pos, feature_names)
        plt.savefig(perm_dir / f"importances_state_{idx}.png")
        plt.close()


def create_gif_from_plots(
    image_filenames, output_filename="animation.gif", duration=0.5
):
    images = [imageio.imread(filename) for filename in image_filenames]
    imageio.mimsave(output_filename, images, duration=duration)


base_path = Path("resources")
env = "CartPole-v0"
estimator = "ensemble.RandomForestClassifier"
seed = str(42)
conf_name = "RandomForest"
desired_return = 200
desired_horizon = 200

path = base_path / env / conf_name
plot_dir = base_path / "example_plot_resources_RF"

res, perm_res = calculate_features_importance(
    path, env, desired_return, desired_horizon, 0.02, 0.02
)
example_plot(res, perm_res, plot_dir)

# Create MDI GIF
image_filenames_mdi = [
    str(plot_dir / "mdi_importance" / f"importances_state_{idx}.png")
    for idx in range(len(res))
]
create_gif_from_plots(image_filenames_mdi, output_filename=str(plot_dir / "mdi_animation.gif"))

# Create Permutation GIF
image_filenames_perm = [
    str(plot_dir / "permutation_importance" / f"importances_state_{idx}.png")
    for idx in range(len(perm_res))
]
create_gif_from_plots(image_filenames_perm, output_filename=str(plot_dir / "permutation_animation.gif"))