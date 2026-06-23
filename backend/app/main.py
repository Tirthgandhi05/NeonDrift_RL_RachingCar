import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.sockets import sio
import socketio
from backend.app.engine import central_loop
from backend.app import state

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Socket.IO to FastAPI app
socket_app = socketio.ASGIApp(sio, app)

@app.on_event("startup")
async def startup_event():
    if state.USE_ONNX:
        print("[server] Starting with ONNX Acceleration ENABLED.")
    else:
        print("[server] Starting with standard PyTorch backend.")
        
    from backend.app.inference.model_manager import load_global_vec_normalize, load_pytorch_model
    from backend.app.inference.onnx_batch_policy import BatchedONNXPolicy
    from backend.app.env.track_pool import TrackPool
    import os
    
    # Initialize the track pool (100 validated tracks)
    TrackPool.initialize(100)
    
    state.MODEL_TYPE = "A2C"
    state.vec_normalizer = load_global_vec_normalize(state.MODEL_TYPE)
    if state.USE_ONNX:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        onnx_path = os.path.join(BASE_DIR, "models", f"{state.MODEL_TYPE.lower()}_policy.onnx")
        state.model = BatchedONNXPolicy(onnx_path, state.MODEL_TYPE, use_gpu=False)
    else:
        state.model = load_pytorch_model(state.MODEL_TYPE)
        
    print(f"[server] Pre-loaded {state.MODEL_TYPE} model.")
        
    asyncio.create_task(central_loop(sio))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:socket_app", host="0.0.0.0", port=8000, reload=True)
