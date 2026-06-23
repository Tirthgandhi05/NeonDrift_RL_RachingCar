import os
import numpy as np
from backend.app.inference.onnx_batch_policy import BatchedONNXPolicy
from backend.app.inference.model_manager import load_global_vec_normalize, load_pytorch_model

def test_onnx_batching():
    model_type = "A2C"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(base_dir, "models", f"{model_type.lower()}_policy.onnx")
    
    onnx_model = BatchedONNXPolicy(onnx_path, model_type, use_gpu=False)
    pt_model = load_pytorch_model(model_type)
    
    # Create a batch of 2 observations
    obs_batch = np.random.randn(2, 11).astype(np.float32)
    
    # Run batch size 1
    onnx_1_0, _ = onnx_model.predict(obs_batch[0:1])
    onnx_1_1, _ = onnx_model.predict(obs_batch[1:2])
    
    # Run batch size 2
    onnx_2, _ = onnx_model.predict(obs_batch)
    
    print("ONNX Batch size 1 results:")
    print(onnx_1_0)
    print(onnx_1_1)
    
    print("\nONNX Batch size 2 results:")
    print(onnx_2)
    
    # Run PyTorch for sanity
    pt_2, _ = pt_model.predict(obs_batch, deterministic=True)
    print("\nPyTorch Batch size 2 results:")
    print(pt_2)

if __name__ == "__main__":
    test_onnx_batching()
