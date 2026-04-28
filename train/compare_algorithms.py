"""
NeonDrift — Algorithm Comparison Script.

Loads trained PPO, A2C, and DQN models and evaluates each on 20 random
tracks.  Produces a comparison table and saves a matplotlib bar chart
to ``comparison_results.png``.

Metrics collected per algorithm:
    • Average reward per step
    • Average steps survived (before crash or truncation)
    • Average speed
    • Average lap progress (%)
    • Crash rate (% of episodes ending in collision)

Usage:
    python train/compare_algorithms.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, DQN

from env.neondrift_env import NeonDriftEnv, DiscreteActionWrapper

N_EVAL_EPISODES = 20
MODEL_DIR = "./models/"

# ─────────────────── Model Loading ─────────────────────────

ALGORITHMS = {
    "PPO": {
        "class": PPO,
        "paths": [
            MODEL_DIR + "ppo_best/best_model",
            MODEL_DIR + "ppo_final",
        ],
        "discrete": False,
    },
    "A2C": {
        "class": A2C,
        "paths": [
            MODEL_DIR + "a2c_best/best_model",
            MODEL_DIR + "a2c_final",
        ],
        "discrete": False,
    },
    "DQN": {
        "class": DQN,
        "paths": [
            MODEL_DIR + "dqn_best/best_model",
            MODEL_DIR + "dqn_final",
        ],
        "discrete": True,
    },
}


def load_model(algo_name: str):
    """Try to load a model for the given algorithm name."""
    cfg = ALGORITHMS[algo_name]
    cls = cfg["class"]
    for path in cfg["paths"]:
        zip_path = path if path.endswith(".zip") else path + ".zip"
        if os.path.isfile(zip_path) or os.path.isfile(path):
            print(f"  [{algo_name}] Loading from: {path}")
            return cls.load(path)
    return None


def evaluate_model(model, algo_name: str, n_episodes: int = N_EVAL_EPISODES):
    """Run the model for n_episodes and return collected metrics."""
    is_discrete = ALGORITHMS[algo_name]["discrete"]

    results = {
        "rewards_per_step": [],
        "steps_survived": [],
        "avg_speeds": [],
        "progress_pcts": [],
        "crashes": 0,
    }

    for ep in range(n_episodes):
        if is_discrete:
            env = DiscreteActionWrapper(NeonDriftEnv())
        else:
            env = NeonDriftEnv()

        obs, info = env.reset()
        total_reward = 0.0
        total_speed = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_speed += info["speed"]
            steps += 1
            done = terminated or truncated

        if terminated and not truncated:
            results["crashes"] += 1

        results["rewards_per_step"].append(total_reward / max(steps, 1))
        results["steps_survived"].append(steps)
        results["avg_speeds"].append(total_speed / max(steps, 1))
        results["progress_pcts"].append(info.get("progress_pct", 0.0))

        env.close()

    return results


def print_comparison(all_results: dict):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("  NeonDrift — Algorithm Comparison Results")
    print(f"  Evaluated over {N_EVAL_EPISODES} random tracks each")
    print("=" * 80)

    header = f"{'Metric':<25} {'PPO':>12} {'A2C':>12} {'DQN':>12}"
    print(header)
    print("-" * 65)

    metrics = [
        ("Avg Reward/Step", "rewards_per_step", "{:.3f}"),
        ("Avg Steps Survived", "steps_survived", "{:.0f}"),
        ("Avg Speed (u/s)", "avg_speeds", "{:.2f}"),
        ("Avg Lap Progress (%)", "progress_pcts", "{:.1f}"),
        ("Crash Rate (%)", None, None),  # special
    ]

    for label, key, fmt in metrics:
        row = f"{label:<25}"
        for algo in ["PPO", "A2C", "DQN"]:
            if algo not in all_results:
                row += f"{'N/A':>12}"
                continue
            r = all_results[algo]
            if key is not None:
                val = np.mean(r[key])
                row += f"{fmt.format(val):>12}"
            else:
                crash_pct = (r["crashes"] / N_EVAL_EPISODES) * 100
                row += f"{crash_pct:>11.0f}%"
        print(row)

    print("=" * 80)

    # Determine winner
    available = [a for a in ["PPO", "A2C", "DQN"] if a in all_results]
    if available:
        best = max(available, key=lambda a: np.mean(all_results[a]["rewards_per_step"]))
        print(f"\n  🏆 Best overall: {best} (highest avg reward/step)")


def plot_comparison(all_results: dict, save_path: str = "comparison_results.png"):
    """Generate and save a bar chart comparing the algorithms."""
    available = [a for a in ["PPO", "A2C", "DQN"] if a in all_results]
    if not available:
        print("No results to plot.")
        return

    metrics = {
        "Avg Reward/Step": [np.mean(all_results[a]["rewards_per_step"]) for a in available],
        "Avg Steps Survived": [np.mean(all_results[a]["steps_survived"]) for a in available],
        "Avg Speed (u/s)": [np.mean(all_results[a]["avg_speeds"]) for a in available],
        "Lap Progress (%)": [np.mean(all_results[a]["progress_pcts"]) for a in available],
        "Crash Rate (%)": [(all_results[a]["crashes"] / N_EVAL_EPISODES) * 100 for a in available],
    }

    colors = {"PPO": "#00FFFF", "A2C": "#FF00AA", "DQN": "#FFFF00"}
    bar_colors = [colors.get(a, "#FFFFFF") for a in available]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle("NeonDrift — Algorithm Comparison (PPO vs A2C vs DQN)",
                 fontsize=14, fontweight="bold", color="white")
    fig.patch.set_facecolor("#0A0A14")

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        bars = ax.bar(available, values, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax.set_title(metric_name, fontsize=10, color="white")
        ax.set_facecolor("#0A0A14")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9, color="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor="#0A0A14", bbox_inches="tight")
    print(f"\n  📊 Chart saved to: {save_path}")


# ─────────────────── Main ─────────────────────────────────

def main():
    print("=" * 60)
    print("  NeonDrift — Algorithm Comparison")
    print(f"  Episodes per algorithm: {N_EVAL_EPISODES}")
    print("=" * 60)

    all_results = {}

    for algo_name in ["PPO", "A2C", "DQN"]:
        print(f"\n--- Evaluating {algo_name} ---")
        model = load_model(algo_name)
        if model is None:
            print(f"  [{algo_name}] No trained model found. Skipping.")
            continue
        results = evaluate_model(model, algo_name)
        all_results[algo_name] = results
        print(f"  [{algo_name}] Done. Avg reward/step: {np.mean(results['rewards_per_step']):.3f}")

    if not all_results:
        print("\nNo models found! Train at least one model first:")
        print("  python train/train_ppo.py")
        print("  python train/train_a2c.py")
        print("  python train/train_dqn.py")
        return

    print_comparison(all_results)
    plot_comparison(all_results)


if __name__ == "__main__":
    main()
