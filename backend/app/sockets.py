import os
import socketio
import asyncio
from backend.app import state
from backend.app.env.neondrift_env import NeonDriftEnv, DiscreteActionWrapper

def is_discrete(m_type):
    return m_type == "DQN"

def make_env(m_type):
    base = NeonDriftEnv()
    if is_discrete(m_type):
        base = DiscreteActionWrapper(base)
    return base

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

@sio.event
async def connect(sid, environ):
    try:
        env = make_env(state.MODEL_TYPE or "PPO")
        obs, info = env.reset()
        
        await sio.emit("track_init", env.get_track_data(), to=sid)
        
        state.clients[sid] = {
            "obs": obs,
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "info": info,
            "env": env,
            "prev_action": None,
            "model_type": state.MODEL_TYPE or "PPO"
        }
        state.active_clients.add(sid)
        print(f"[server] Client {sid} connected. Total clients: {len(state.clients)}")
    except Exception as e:
        print(f"[server] Error setting up env for {sid}: {e}")
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    if sid in state.active_clients:
        state.active_clients.remove(sid)
    if sid in state.clients:
        del state.clients[sid]
    print(f"Client disconnected: {sid}. Total clients: {len(state.clients)}")

@sio.event
async def reset_env(sid, data):
    if sid in state.clients:
        try:
            obs, info = state.clients[sid]["env"].reset()
            
            await sio.emit("track_init", state.clients[sid]["env"].get_track_data(), to=sid)
            
            state.clients[sid]["obs"] = obs
            state.clients[sid]["reward"] = 0.0
            state.clients[sid]["terminated"] = False
            state.clients[sid]["truncated"] = False
            state.clients[sid]["info"] = info
            await sio.emit("reset_complete", {"obs": obs.tolist()}, to=sid)
        except Exception as e:
            print(f"[server] Error resetting env for {sid}: {e}")

@sio.event
async def set_model(sid, data):
    from backend.app.inference.model_manager import load_global_vec_normalize, load_pytorch_model
    from backend.app.inference.onnx_batch_policy import BatchedONNXPolicy
    
    new_model = data.get("model", "PPO").upper()
    print(f"[server] Client {sid} requested model change to {new_model}")
    
    if new_model != state.MODEL_TYPE:
        state.MODEL_TYPE = new_model
        state.vec_normalizer = load_global_vec_normalize(state.MODEL_TYPE)
        
        if state.USE_ONNX:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            onnx_path = os.path.join(BASE_DIR, "models", f"{state.MODEL_TYPE.lower()}_policy.onnx")
            import asyncio
            state.model = await asyncio.to_thread(BatchedONNXPolicy, onnx_path, state.MODEL_TYPE, False)
        else:
            import asyncio
            state.model = await asyncio.to_thread(load_pytorch_model, state.MODEL_TYPE)
            
        print(f"[server] Switched to {state.MODEL_TYPE}")
        await sio.emit("model_changed", {"model": state.MODEL_TYPE})

        # Update model type for all existing active clients so they use the new loop
        for c in state.clients.values():
            c["model_type"] = state.MODEL_TYPE
            
@sio.event
async def manual_control(sid, data):
    pass
