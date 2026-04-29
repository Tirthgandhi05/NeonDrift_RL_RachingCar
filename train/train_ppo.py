"""
NeonDrift — PPO Training Script.

PPO (Proximal Policy Optimisation) is the recommended algorithm and the
primary model served by the inference server.

Key properties:
    • Continuous action space (Box(2))
    • Trust-region clipping (clip_range=0.2)  →  stable monotonic improvement
    • Best zero-shot generalisation to unseen tracks

Ablation note:
    Compared to A2C, PPO adds an epsilon-clipping constraint that prevents
    excessively large policy updates.  This is the main reason PPO achieves
    more stable training curves and better generalisation.
"""

import sys
import os

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from env.neondrift_env import NeonDriftEnv

# ─────────────────── Shared Constants ──────────────────────
TOTAL_TIMESTEPS = 1_000_000
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models")
LOG_PATH = os.path.join(PROJECT_ROOT, "logs")
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5


def main():
    # Vectorised training env (4 parallel workers)
    env = make_vec_env(lambda: NeonDriftEnv(), n_envs=4)

    # Single evaluation env
    eval_env = NeonDriftEnv()

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,          # ε = 0.2 — the trust-region clipping parameter
        ent_coef=0.005,          # Slightly less exploration (reward is clearer now)
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=LOG_PATH,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_SAVE_PATH, "ppo_best"),
        log_path=os.path.join(LOG_PATH, "ppo_eval"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
    )

    print("=" * 60)
    print("  NeonDrift - PPO Training")
    print(f"  Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"  Clip range (e)  : 0.2")
    print(f"  Vectorised envs : 4")
    print("=" * 60)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    model.save(os.path.join(MODEL_SAVE_PATH, "ppo_final"))
    print("PPO training complete. Model saved.")


if __name__ == "__main__":
    main()
