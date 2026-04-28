"""
NeonDrift — Model Loader.

Loads a saved Stable Baselines3 model with support for multiple
algorithm types.  Default is PPO with a fallback chain:
    1. Try ``models/ppo_best/best_model``  (EvalCallback output)
    2. Fall back to ``models/ppo_final``   (final checkpoint)
    3. Raise a clear error if neither exists.

For comparison evaluations, use ``load_model(model_type="A2C")`` or
``load_model(model_type="DQN")`` to load those models instead.

DQN Note:
    DQN uses a discrete action space.  After loading, the caller is
    responsible for wrapping the environment with DiscreteActionWrapper.
    Use ``is_discrete(model_type)`` to check whether wrapping is needed.
"""

import os
from stable_baselines3 import PPO, A2C, DQN

MODEL_CLASSES = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
}

# Per-algorithm default paths (primary, fallback)
MODEL_PATHS = {
    "PPO": ("./models/ppo_best/best_model", "./models/ppo_final"),
    "A2C": ("./models/a2c_best/best_model", "./models/a2c_final"),
    "DQN": ("./models/dqn_best/best_model", "./models/dqn_final"),
}

# Algorithms that require DiscreteActionWrapper
DISCRETE_ALGORITHMS = {"DQN"}


def is_discrete(model_type: str) -> bool:
    """Return True if the algorithm requires a DiscreteActionWrapper."""
    return model_type.upper() in DISCRETE_ALGORITHMS


def load_model(
    primary_path: str | None = None,
    fallback_path: str | None = None,
    model_type: str = "PPO",
):
    """
    Load and return a trained model.

    Parameters
    ----------
    primary_path : str | None
        Preferred model path (EvalCallback best model).  Defaults to the
        standard path for *model_type* when None.
    fallback_path : str | None
        Fallback model path (final training checkpoint).  Defaults to the
        standard path for *model_type* when None.
    model_type : str
        Algorithm type: "PPO", "A2C", or "DQN".

    Returns
    -------
    BaseAlgorithm
        The loaded Stable Baselines3 model.

    Raises
    ------
    FileNotFoundError
        If neither path resolves to a valid model file.
    ValueError
        If model_type is not recognised.
    """
    model_type = model_type.upper()

    if model_type not in MODEL_CLASSES:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from: {list(MODEL_CLASSES.keys())}"
        )

    cls = MODEL_CLASSES[model_type]

    # Fall back to per-algorithm default paths when not explicitly provided
    defaults = MODEL_PATHS[model_type]
    resolved_primary = primary_path if primary_path is not None else defaults[0]
    resolved_fallback = fallback_path if fallback_path is not None else defaults[1]

    # SB3 appends .zip if not present, so check both variants
    for path in (resolved_primary, resolved_fallback):
        zip_path = path if path.endswith(".zip") else path + ".zip"
        if os.path.isfile(zip_path) or os.path.isfile(path):
            print(f"[model_loader] Loading {model_type} model from: {path}")
            return cls.load(path)

    raise FileNotFoundError(
        f"No trained {model_type} model found.\n"
        f"  Checked: {resolved_primary}\n"
        f"  Checked: {resolved_fallback}\n"
        f"Run  python train/train_{model_type.lower()}.py  first."
    )
