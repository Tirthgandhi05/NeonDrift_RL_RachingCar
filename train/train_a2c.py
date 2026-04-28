"""
NeonDrift — A2C Training Script.

A2C (Advantage Actor-Critic) uses a continuous action space like PPO but
does NOT apply trust-region clipping.

Key properties:
    • Continuous action space (Box(2))
    • No clip_range, no n_epochs, no batch_size
    • Learns faster than DQN initially but may exhibit unstable policy
      updates (catastrophic forgetting on unseen tracks)

Ablation note:
    A2C is the "catastrophic forgetting baseline."  Without the clipping
    constraint, large policy updates can destroy performance on track
    layouts seen earlier in training.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from env.neondrift_env import NeonDriftEnv

# ─────────────────── Shared Constants ──────────────────────
TOTAL_TIMESTEPS = 1_000_000
MODEL_SAVE_PATH = "./models/"
LOG_PATH = "./logs/"
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5


def main():
    env = make_vec_env(lambda: NeonDriftEnv(), n_envs=4)
    eval_env = NeonDriftEnv()

    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=7e-4,
        n_steps=16,              # Increased from 5 → 16 (less noisy updates)
        gamma=0.99,
        gae_lambda=0.95,         # Changed from 1.0 → 0.95 (proper GAE)
        ent_coef=0.01,           # Changed from 0.0 → 0.01 (prevents premature convergence)
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=LOG_PATH,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_SAVE_PATH + "a2c_best/",
        log_path=LOG_PATH + "a2c_eval/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
    )

    print("=" * 60)
    print("  NeonDrift - A2C Training")
    print(f"  Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"  No clip_range   : catastrophic forgetting baseline")
    print(f"  Vectorised envs : 4")
    print("=" * 60)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    model.save(MODEL_SAVE_PATH + "a2c_final")
    print("A2C training complete. Model saved.")


if __name__ == "__main__":
    main()
