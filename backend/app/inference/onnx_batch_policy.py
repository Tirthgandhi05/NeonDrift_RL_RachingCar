import os
import numpy as np
import onnxruntime as ort

# kill BLAS oversubscription at OS level too — session options alone aren't enough
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

class BatchedONNXPolicy:
    def __init__(self, onnx_path: str, model_type: str, use_gpu: bool = False):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.model_type = model_type.upper()

    def predict(self, batch_obs: np.ndarray, deterministic: bool = True):
        batch_obs = np.ascontiguousarray(batch_obs, dtype=np.float32)
        raw = self.session.run(self.output_names, {self.input_name: batch_obs})[0]

        if self.model_type == "DQN":
            actions = np.argmax(raw, axis=1)          # (N,) discrete
        else:
            actions = np.clip(raw, -1.0, 1.0)         # (N, act_dim) continuous

        return actions, None  # match SB3 .predict() signature
