import asyncio
import numpy as np
import math
from backend.app import state

def normalize_batch(batch_obs):
    if state.vec_normalizer is None:
        return batch_obs
    return np.clip(
        (batch_obs - state.vec_normalizer.obs_rms.mean) / np.sqrt(state.vec_normalizer.obs_rms.var + state.vec_normalizer.epsilon),
        -state.vec_normalizer.clip_obs, state.vec_normalizer.clip_obs
    ).astype(np.float32)

def compute_lidar_ray_endpoints(info):
    car_x = info.get("car_x", 0.0)
    car_y = info.get("car_y", 0.0)
    car_heading = info.get("car_heading", 0.0)
    # Scale back the normalized lidar readings for rendering
    lidar_readings = np.array(info.get("lidar", [])) * 200.0 # MAX_RAY_LEN
    
    lidar_rays = []
    num_rays = len(lidar_readings)
    if num_rays > 0:
        angles = np.linspace(-math.pi/2, math.pi/2, num_rays)
        for i, dist in enumerate(lidar_readings):
            ray_angle = car_heading + angles[i]
            end_x = car_x + math.cos(ray_angle) * dist
            end_y = car_y + math.sin(ray_angle) * dist
            lidar_rays.append({
                "angle": float(angles[i]),
                "distance": float(dist),
                "end_x": float(end_x),
                "end_y": float(end_y)
            })
    return lidar_rays

async def central_loop(sio):
    TICK = 1 / 30.0
    while True:
        t0 = asyncio.get_event_loop().time()
        
        active = {sid: c for sid, c in state.clients.items() if sid in state.active_clients and state.model is not None and c["model_type"] == state.MODEL_TYPE}

        if active:
            sids = list(active.keys())
            batch_obs = np.stack([active[s]["obs"] for s in sids])
            batch_obs_norm = normalize_batch(batch_obs)

            actions = await asyncio.to_thread(state.model.predict, batch_obs_norm, deterministic=True)
            if isinstance(actions, tuple):
                actions = actions[0]

            emit_tasks = []
            
            for i, sid in enumerate(sids):
                c = active[sid]
                a = actions[i]
                if isinstance(a, np.ndarray) and a.dtype.kind in 'fc':
                    if c.get("prev_action") is not None:
                        a = 0.7 * a + 0.3 * c["prev_action"]
                    c["prev_action"] = a

                obs, reward, terminated, truncated, info = c["env"].step(a)
                c["obs"] = obs
                c["info"] = info

                lidar_rays = compute_lidar_ray_endpoints(info)
                payload = {
                    "car_x": float(info["car_x"]),
                    "car_y": float(info["car_y"]),
                    "car_heading": float(info["car_heading"]),
                    "speed": float(info["speed"]),
                    "reward": float(reward),
                    "lidar_rays": lidar_rays,
                    "state_vector": obs.tolist(),
                    "progress_pct": float(info.get("progress_pct", 0.0)),
                }
                
                emit_tasks.append(sio.emit("telemetry", payload, to=sid))

                if terminated or truncated:
                    obs, info = c["env"].reset()
                    c["obs"] = obs
                    c["info"] = info
                    c["prev_action"] = None
                    emit_tasks.append(sio.emit("reset", {}, to=sid))
                    emit_tasks.append(sio.emit("track_init", c["env"].get_track_data(), to=sid))

            # Fire all telemetry emits concurrently
            await asyncio.gather(*emit_tasks)

        elapsed = asyncio.get_event_loop().time() - t0
        await asyncio.sleep(max(0, TICK - elapsed))
