"""
NeonDrift — FastAPI + Socket.IO Inference Server.

Loads a trained model and streams telemetry to connected React frontends
at ~30 FPS via WebSocket (Socket.IO protocol).

Supported algorithms (set via MODEL_TYPE env var):
    PPO  — default, continuous action space
    A2C  — continuous action space
    DQN  — discrete action space (DiscreteActionWrapper applied automatically)

Usage:
    python inference/server.py                  # PPO (default)
    MODEL_TYPE=DQN python inference/server.py   # DQN
    MODEL_TYPE=A2C python inference/server.py   # A2C

Endpoints:
    GET  /health   →  {"status": "ok", "model": "<MODEL_TYPE>"}
    GET  /reset    →  resets the environment
    WS   connect   →  starts the 30 FPS game loop
"""

import sys
import os
import asyncio

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import socketio
import uvicorn
from fastapi import FastAPI

from env.neondrift_env import NeonDriftEnv, DiscreteActionWrapper, MAX_RAY_LEN, MAX_SPEED
from inference.model_loader import load_model, is_discrete
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ─────────────────── Socket.IO + FastAPI ──────────────────
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = FastAPI(title="NeonDrift Inference Server")
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# ─────────────────── Load Model & Env ─────────────────────
MODEL_TYPE = os.environ.get("MODEL_TYPE", "PPO").upper()

model = load_model(model_type=MODEL_TYPE)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# DQN requires a discrete action wrapper; PPO/A2C use the raw env
def make_env():
    base = NeonDriftEnv()
    if is_discrete(MODEL_TYPE):
        base = DiscreteActionWrapper(base)
    
    # Apply VecNormalize if stats exist (needed for A2C)
    vec_path = os.path.join(BASE_DIR, "models", f"{MODEL_TYPE.lower()}_vecnormalize.pkl")
    if os.path.exists(vec_path):
        print(f"[server] Loading VecNormalize stats from: {vec_path}")
        vec_env = DummyVecEnv([lambda: base])
        vec_env = VecNormalize.load(vec_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        return vec_env
    return base

env = make_env()
obs = env.reset()
if isinstance(obs, tuple):  # Gymnasium raw env returns (obs, info)
    obs, info = obs
else:
    info = env.get_attr("_get_info")[0]() if hasattr(env, "get_attr") else {}

print(f"[server] Algorithm : {MODEL_TYPE}")
print(f"[server] Discrete  : {is_discrete(MODEL_TYPE)}")


# ────────────── LiDAR Ray Endpoints Helper ────────────────
def compute_lidar_ray_endpoints(info: dict) -> list:
    """
    Returns list of 7 dicts: {x1, y1, x2, y2, hit_fraction}
    where (x1,y1) is car position and (x2,y2) is ray endpoint.
    hit_fraction is the normalized distance (0=wall, 1=max range).
    """
    rays = []
    angles = np.linspace(-np.pi / 2, np.pi / 2, 7)
    for i, angle_offset in enumerate(angles):
        ray_angle = info["car_heading"] + angle_offset
        hit_dist = info["lidar"][i] * MAX_RAY_LEN  # denormalize
        x2 = info["car_x"] + hit_dist * np.cos(ray_angle)
        y2 = info["car_y"] + hit_dist * np.sin(ray_angle)
        rays.append({
            "x1": float(info["car_x"]),
            "y1": float(info["car_y"]),
            "x2": float(x2),
            "y2": float(y2),
            "hit_fraction": float(info["lidar"][i]),
        })
    return rays


# ────────────────── HTTP Endpoints ────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_TYPE}


@app.get("/reset")
async def reset_env():
    global obs, info
    obs, info = env.reset()
    return {"status": "reset"}


# ──────────────── Socket.IO Events ────────────────────────
@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")
    asyncio.create_task(game_loop(sid))


@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")


async def game_loop(sid):
    """30 FPS game loop: predict → step → emit telemetry."""
    global obs, info
    obs = env.reset()
    if isinstance(obs, tuple):
        obs, info = obs
    else:
        info = env.get_attr("_get_info")[0]() if hasattr(env, "get_attr") else {}

    while True:
        # Check if client is still connected
        try:
            # Handle VecEnv arrays vs single env scalar returns
            if hasattr(env, "get_attr"):  # It's a VecEnv
                action, _ = model.predict(obs, deterministic=True)
                obs, reward_arr, done_arr, info_arr = env.step(action)
                reward = reward_arr[0]
                terminated = done_arr[0]
                truncated = False # VecEnv doesn't expose truncated separately
                info = info_arr[0]
            else:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

            lidar_rays = compute_lidar_ray_endpoints(info)

            payload = {
                "car_x": float(info["car_x"]),
                "car_y": float(info["car_y"]),
                "car_heading": float(info["car_heading"]),
                "speed": float(info["speed"]),
                "reward": float(reward),
                "lidar_readings": info["lidar"],
                "lidar_rays": lidar_rays,
                "left_boundary": info["left_boundary"],
                "right_boundary": info["right_boundary"],
                "centerline": info["centerline"],
                "state_vector": obs[0].tolist() if hasattr(env, "get_attr") else obs.tolist(),
                "progress_pct": float(info.get("progress_pct", 0.0)),
            }

            await sio.emit("telemetry", payload, to=sid)

            if terminated or truncated:
                obs = env.reset()
                if isinstance(obs, tuple): obs = obs[0]
                await sio.emit("reset", {}, to=sid)

            await asyncio.sleep(1 / 30)  # 30 FPS

        except Exception as e:
            print(f"Game loop error for {sid}: {e}")
            break


# ──────────────────── Entry Point ─────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "inference.server:socket_app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
