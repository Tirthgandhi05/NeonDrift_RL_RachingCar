"""
NeonDrift — ONNX Export Script

Loads trained Stable Baselines 3 policies (PPO, A2C, DQN) and extracts their 
deterministic action networks into lightweight .onnx graphs.

Usage:
    python train/export_onnx.py
"""

import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.model_loader import load_model

class OnnxablePolicy(torch.nn.Module):
    """Wrapper to extract the deterministic action network from PPO/A2C ActorCritic policies."""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        features = self.policy.extract_features(observation)
        latent_pi, _ = self.policy.mlp_extractor(features)
        return self.policy.action_net(latent_pi)

class OnnxableDQNPolicy(torch.nn.Module):
    """Wrapper to extract the best action (argmax Q) from a DQN policy."""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        q_values = self.policy.q_net(observation)
        return torch.argmax(q_values, dim=1)

def export_model(model_type: str, output_path: str):
    print(f"Exporting {model_type} to ONNX...")
    try:
        model = load_model(model_type=model_type)
    except FileNotFoundError:
        print(f"  [!] No trained {model_type} model found. Skipping.")
        return

    # NeonDrift observation space is 11 float32 values
    dummy_input = torch.randn(1, 11)

    if model_type in ["PPO", "A2C"]:
        onnxable_model = OnnxablePolicy(model.policy)
    else:  # DQN
        onnxable_model = OnnxableDQNPolicy(model.policy)

    onnxable_model.eval()

    # Ensure models directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        onnxable_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"  [+] Successfully exported to: {output_path}")

def main():
    print("=" * 50)
    print("  NeonDrift ONNX Exporter")
    print("=" * 50)
    
    export_model("PPO", "models/ppo_policy.onnx")
    export_model("A2C", "models/a2c_policy.onnx")
    export_model("DQN", "models/dqn_policy.onnx")

if __name__ == "__main__":
    main()
