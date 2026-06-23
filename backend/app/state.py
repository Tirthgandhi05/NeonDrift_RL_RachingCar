import os

# Globals
clients = {}
active_clients = set()
MODEL_TYPE = None
model = None
vec_normalizer = None
USE_ONNX = os.environ.get("USE_ONNX", "0") == "1"
