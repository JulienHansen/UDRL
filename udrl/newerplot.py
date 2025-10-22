from udrl.policies import SklearnPolicy
from udrl.agent import UpsideDownAgent, AgentHyper
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from itertools import zip_longest, tee
from tqdm import tqdm, trange
import imageio
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def calculate_local_mdi_importance(agent, ext_state):
    """Calculates local MDI importance for a Random Forest Classifier."""
    feature_importances = {}
    for t in agent.policy.estimator.estimators_:
        branch = np.array(t.decision_path(ext_state).todense(), dtype=bool)
        imp = t.tree_.impurity[branch[0]]
        for f, i in zip(
            t.tree_.feature[branch[0]][:-1], imp[:-1] - imp[1:]
        ):
            feature_importances.setdefault(f, []).append(i)

    summed_importances = [
        sum(feature_importances.get(k, []))
        for k in range(ext_state.shape[1])
    ]
    return summed_importances


def calculate_local_permutation_importance(agent, ext_state):
    """Calculates local permutation importance for any classifier."""
    base_prediction = agent.policy.estimator.predict(ext_state)
    local_importances = []
    
    for i in range(ext_state.shape[1]):
        perturbed_state = ext_state.copy()
        perturbed_state[:, i] += np.random.normal(0, 0.1) 
        
        perturbed_prediction = agent.policy.estimator.predict(perturbed_state)
        
        # Importance is the change in the model's prediction
        importance = np.sum(base_prediction != perturbed_prediction)
        local_importances.append(importance)
        
    return local_importances


def calculate_ep_importances(
    episode, agent, desired_return, desired_horizon
):
    """Calculates all importance scores for a single episode."""
    ep_mdi_features = []
    ep_perm_features = []

    for state, _, reward in zip(*episode.values()):
        command = np.array(
            [
                desired_return * agent.conf.return_scale,
                desired_horizon * agent.conf.horizon_scale,
            ]
        )
        command = np.expand_dims(command, axis=0)
        ext_state = np.concatenate((state, command), axis=1)

        # MDI Importance (Only for Random Forest)
        if isinstance(agent.policy.estimator, RandomForestClassifier):
            mdi_imp = calculate_local_mdi_importance(agent, ext_state)
            ep_mdi_features.append(mdi_imp)

        # Permutation Importance (for all classifiers)
        perm_imp = calculate_local_permutation_importance(agent, ext_state)
        ep_perm_features.append(perm_imp)
        
        desired_return -= reward
        desired_horizon = max(desired_horizon - 1, 1)

    return ep_mdi_features, ep_perm_features


def summarize_episodes_feat(episodes_feat, summarize_funs=[np.mean, np.std]):
    """Summarizes feature importances across episodes."""
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


def collect_data(
    path: Path, env: str, desired_return: int, desired_horizon: int, 
    horizon_scale: float, return_scale: float, redundancy: int = 100
):
    """Collects data and loads the agent."""
    policy = SklearnPolicy.load(str(path / "policy"))
    hyper = AgentHyper(
        env, warm_up=0, horizon_scale=horizon_scale, return_scale=return_scale
    )
    agent = UpsideDownAgent(hyper, policy)

    for _ in trange(redundancy, desc="Collect Data"):
        agent.collect_episode(desired_return, desired_horizon, test=True)

    episodes = agent.memory.buffer
    states = np.concatenate([ep['states'] for ep in episodes])
    commands = np.concatenate([ep['commands'] for ep in episodes])
    X = np.concatenate((states, commands), axis=1)
    
    return agent, episodes, X


def calculate_all_importances(agent, episodes, desired_return, desired_horizon):
    """Calculates all importance scores for all episodes."""
    ep_mdi_results = []
    ep_perm_results = []

    for ep in tqdm(episodes, desc="Calculate importance features"):
        mdi_imp, perm_imp = calculate_ep_importances(
            ep, agent, desired_return, desired_horizon
        )
        if mdi_imp:
            ep_mdi_results.append(mdi_imp)
        ep_perm_results.append(perm_imp)

    mdi_importances = summarize_episodes_feat(ep_mdi_results) if ep_mdi_results else None
    perm_importances = summarize_episodes_feat(ep_perm_results)
    
    return mdi_importances, perm_importances


def plot_importances(
    mdi_importances, perm_importances, global_mdi_importance, plot_dir, estimator
):
    """Plots all importance scores."""
    feature_names = ["x", "ẋ", "θ", "θ*", r"$d_t^r$", r"$d_t^h$"]

    # Plot Gini/MDI Importance
    if isinstance(estimator, RandomForestClassifier):
        mdi_dir = plot_dir / "mdi_importance"
        mdi_dir.mkdir(parents=True, exist_ok=True)
        for idx, state_feat in tqdm(
            enumerate(mdi_importances), desc="Plotting MDI", total=len(mdi_importances)
        ):
            x_pos = np.arange(len(state_feat))
            plt.figure()
            plt.title(f"{estimator.__class__.__name__} State {idx}")
            plt.bar(x_pos, [x[0] for x in state_feat], yerr=[x[1] for x in state_feat])
            plt.xticks(x_pos, feature_names)
            plt.savefig(mdi_dir / f"importances_state_{idx}.png")
            plt.close()
    
    # Plot global MDI for Gradient Boosting
    if isinstance(estimator, GradientBoostingClassifier):
        global_mdi_dir = plot_dir / "global_mdi_importance"
        global_mdi_dir.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.title(f"Global MDI for {estimator.__class__.__name__}")
        plt.bar(np.arange(len(global_mdi_importance)), global_mdi_importance)
        plt.xticks(np.arange(len(global_mdi_importance)), feature_names)
        plt.savefig(global_mdi_dir / "global_mdi_plot.png")
        plt.close()

    # Plot Permutation Importance
    perm_dir = plot_dir / "permutation_importance"
    perm_dir.mkdir(parents=True, exist_ok=True)
    for idx, perm_feat in tqdm(
        enumerate(perm_importances), desc="Plotting Permutation", total=len(perm_importances)
    ):
        x_pos = np.arange(len(perm_feat))
        plt.figure()
        plt.title(f"{estimator.__class__.__name__} State {idx}")
        plt.bar(x_pos, [x[0] for x in perm_feat], yerr=[x[1] for x in perm_feat])
        plt.xticks(x_pos, feature_names)
        plt.savefig(perm_dir / f"importances_state_{idx}.png")
        plt.close()


def pdp_plot(agent, X, plot_dir):
    """Creates Partial Dependence Plots."""
    pdp_dir = plot_dir / "pdp_plots"
    pdp_dir.mkdir(parents=True, exist_ok=True)
    
    feature_names = ["x", "ẋ", "θ", "θ*", r"$d_t^r$", r"$d_t^h$"]
    features = [0, 1, 2, 3]

    for i in tqdm(features, desc="Plotting PDPs"):
        display = PartialDependenceDisplay.from_estimator(
            agent.policy.estimator, X, [i], feature_names=feature_names,
            percentiles=(0.0, 1.0)
        )
        fig = display.figure_
        ax = display.axes_[0]
        ax.set_title(f"PDP for {feature_names[i]}")
        plt.tight_layout()
        plt.savefig(pdp_dir / f"pdp_plot_feature_{i}.png")
        plt.close(fig)


def create_gif_from_plots(image_filenames, output_filename="animation.gif", duration=0.5):
    """Creates a GIF from a list of image filenames."""
    images = [imageio.imread(filename) for filename in image_filenames]
    imageio.mimsave(output_filename, images, duration=duration)


# Main execution block
base_path = Path("data")
env = "CartPole-v0"
estimator = "ensemble.GradientBoostingClassifier"
seed = str(42)
conf_name = "estimator_nameensemble.GradientBoostingClassifier"
desired_return = 200
desired_horizon = 200

# Choose your estimator here to switch between models
# estimator = "RandomForestClassifier"

path = base_path / env / conf_name / seed
plot_dir = base_path / "example_plot4"


# Step 1: Collect Data
agent, episodes, X = collect_data(
    path, env, desired_return, desired_horizon, 0.02, 0.02
)

# Step 2: Calculate all importances
mdi_res, perm_res = calculate_all_importances(
    agent, episodes, desired_return, desired_horizon
)

# Get the estimator instance
trained_estimator = agent.policy.estimator

# Step 3: Plot all importance scores
if isinstance(trained_estimator, GradientBoostingClassifier):
    global_mdi_imp = trained_estimator.feature_importances_
else:
    global_mdi_imp = None
    
plot_importances(mdi_res, perm_res, global_mdi_imp, plot_dir, trained_estimator)

# Step 4: Create PDP plots
pdp_plot(agent, X, plot_dir)

# Step 5: Create GIFs for permutation importance
image_filenames_perm = [
    str(plot_dir / "permutation_importance" / f"importances_state_{idx}.png")
    for idx in range(len(perm_res))
]
create_gif_from_plots(image_filenames_perm, output_filename=str(plot_dir / "permutation_animation.gif"))