import os
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from backend.app.env.neondrift_env import NeonDriftEnv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def load_global_vec_normalize(m_type):
    vec_path = os.path.join(BASE_DIR, "models", f"{m_type.lower()}_vecnormalize.pkl")
    if os.path.exists(vec_path):
        dummy = DummyVecEnv([lambda: NeonDriftEnv()])
        vn = VecNormalize.load(vec_path, dummy)
        vn.training = False
        vn.norm_reward = False
        return vn
    return None

def load_pytorch_model(model_type):
    model_path = os.path.join(BASE_DIR, "models", f"{model_type.lower()}_final")
    
    if model_type == "PPO":
        model = PPO.load(model_path)
    elif model_type == "A2C":
        model = A2C.load(model_path)
    elif model_type == "DQN":
        model = DQN.load(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model
