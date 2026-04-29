"""
NeonDrift — DQN Training Script.

DQN (Deep Q-Network) is the only algorithm that uses a DISCRETE action
space.  A DiscreteActionWrapper maps 9 integer indices to pre-defined
continuous [steer_delta, throttle] bundles.

Key properties:
    • Discrete actions  →  quantised steering (9 choices)
    • Replay buffer + target network
    • Struggles to find smooth racing lines; converges slowly

Ablation note:
    The coarse discretisation forces jerky behaviour.  This is the
    "quantisation penalty" baseline — demonstrating why continuous
    control (PPO/A2C) is superior for smooth steering tasks.

IMPORTANT: Do NOT use make_vec_env for DQN.  SB3's DQN does not
support vectorised environments.
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

from env.neondrift_env import NeonDriftEnv, DiscreteActionWrapper

# ─────────────────── Shared Constants ──────────────────────
TOTAL_TIMESTEPS = 1_000_000
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models")
LOG_PATH = os.path.join(PROJECT_ROOT, "logs")
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5


def main():
    # Wrap environment for discrete actions — NO make_vec_env!
    env = DiscreteActionWrapper(NeonDriftEnv())
    eval_env = DiscreteActionWrapper(NeonDriftEnv())

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        tensorboard_log=LOG_PATH,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_SAVE_PATH, "dqn_best"),
        log_path=os.path.join(LOG_PATH, "dqn_eval"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
    )

    print("=" * 60)
    print("  NeonDrift - DQN Training")
    print(f"  Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"  Discrete actions: 9 (Hard L/Soft L/Slight L/Straight/Slight R/Soft R/Hard R/Brake/Coast)")
    print(f"  No make_vec_env : single env only")
    print("=" * 60)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    model.save(os.path.join(MODEL_SAVE_PATH, "dqn_final"))
    print("DQN training complete. Model saved.")


if __name__ == "__main__":
    main()
