import os
import time
import numpy as np
from backend.app.inference.onnx_batch_policy import BatchedONNXPolicy

def test_onnx_time():
    model_type = "A2C"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(base_dir, "models", f"{model_type.lower()}_policy.onnx")
    
    onnx_model = BatchedONNXPolicy(onnx_path, model_type, use_gpu=False)
    obs_batch = np.random.randn(20, 11).astype(np.float32)
    
    # warmup
    for _ in range(5):
        onnx_model.predict(obs_batch)
        
    t0 = time.perf_counter()
    for _ in range(100):
        onnx_model.predict(obs_batch)
    t1 = time.perf_counter()
    
    print(f"ONNX Batch size 20: {(t1 - t0) * 10} ms per predict")

if __name__ == "__main__":
    test_onnx_time()
