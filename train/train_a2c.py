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

Hyperparameter rationale:
    n_steps=256      — 1024 transitions/update (4 envs×256) for stable
                       advantage estimates on long racing episodes.
    learning_rate    — linear decay 3e-4→0  prevents late-training
                       oscillations (critical without PPO's clip guard).
    normalize_advantage — reduces gradient variance from reward-scale
                       differences (collision=-10 vs progress≈+1).
    VecNormalize     — online running-mean normalisation of obs & rewards.
    net_arch [128,128] — slightly larger network; minimal overhead for
                       11-dim observations, better value-function fit.
"""

import sys
import os

# ── Resolve project root so model/log paths are absolute ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

from env.neondrift_env import NeonDriftEnv

# ─────────────────── Shared Constants ──────────────────────
TOTAL_TIMESTEPS = 1_000_000
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models")
LOG_PATH = os.path.join(PROJECT_ROOT, "logs")
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5


# ─────────────────── LR Schedule ───────────────────────────
def linear_schedule(initial_lr: float):
    """
    Return a callable that computes a linearly decaying learning rate.

    The returned function maps progress_remaining ∈ [1 → 0] to
    lr ∈ [initial_lr → 0].  SB3 calls this every rollout collection.
    """
    def _schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_lr
    return _schedule


def main():
    # ── Vectorised training env (4 parallel workers) ──
    env = make_vec_env(lambda: NeonDriftEnv(), n_envs=4)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ── Single evaluation env (also normalised) ──
    eval_env = make_vec_env(lambda: NeonDriftEnv(), n_envs=1)
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False,  # don't normalise eval rewards
        clip_obs=10.0,
    )

    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=linear_schedule(3e-4),  # linear decay: stable without clipping
        n_steps=256,             # 4 envs × 256 = 1024 transitions per update
        gamma=0.99,
        gae_lambda=0.95,         # proper GAE (was 1.0 → pure MC before)
        ent_coef=0.01,           # exploration entropy bonus
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,  # critical: reduces gradient variance
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
        ),
        tensorboard_log=LOG_PATH,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_SAVE_PATH, "a2c_best"),
        log_path=os.path.join(LOG_PATH, "a2c_eval"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
    )

    print("=" * 60)
    print("  NeonDrift - A2C Training")
    print(f"  Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"  n_steps         : 256  (1024 transitions/update)")
    print(f"  LR schedule     : linear 3e-4 → 0")
    print(f"  normalize_adv   : True")
    print(f"  VecNormalize    : obs + rewards")
    print(f"  net_arch        : pi=[128,128], vf=[128,128]")
    print(f"  No clip_range   : catastrophic forgetting baseline")
    print(f"  Vectorised envs : 4")
    print("=" * 60)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, log_interval=4)

    # Save the model
    model.save(os.path.join(MODEL_SAVE_PATH, "a2c_final"))

    # Save the VecNormalize statistics (needed for inference)
    env.save(os.path.join(MODEL_SAVE_PATH, "a2c_vecnormalize.pkl"))

    print("A2C training complete. Model saved.")
    print(f"  Model : {os.path.join(MODEL_SAVE_PATH, 'a2c_final.zip')}")
    print(f"  Stats : {os.path.join(MODEL_SAVE_PATH, 'a2c_vecnormalize.pkl')}")


if __name__ == "__main__":
    main()
