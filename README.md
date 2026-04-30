# NeonDrift — Real-Time Autonomous Navigation via Reinforcement Learning

A full-stack reinforcement learning system where a trained agent navigates a 2D procedurally-generated racing track in real time.

## Architecture

| Layer | Description |
|-------|-------------|
| **Python RL Layer** | Custom Gymnasium environment + three training scripts (DQN, A2C, PPO) using Stable Baselines3 |
| **Python Backend** | FastAPI + Socket.IO server streaming telemetry at 30 FPS |
| **React Frontend** | Neon-cyberpunk dashboard with HTML5 canvas rendering and live telemetry graphs |

### Core Constraint

> **No computer vision. No pixels. No CNNs.**
>
> The agent's entire perception comes from a simulated 1D LiDAR — an array of 7 ray distances — plus its own speed, steering angle, track progress, and heading alignment (11-float observation vector).

## Project Structure

```
neondrift/
├── env/
│   ├── __init__.py
│   └── neondrift_env.py          # Custom Gymnasium environment
├── train/
│   ├── train_dqn.py
│   ├── train_a2c.py
│   ├── train_ppo.py
│   └── compare_algorithms.py    # Evaluation & comparison script
├── inference/
│   ├── server.py                 # FastAPI + Socket.IO server
│   └── model_loader.py           # Loads saved SB3 model
├── models/                       # Saved model files (gitignored)
│   └── .gitkeep
├── logs/                         # TensorBoard logs (gitignored)
│   └── .gitkeep
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── RaceCanvas.jsx
│   │   │   └── TelemetryPanel.jsx
│   │   ├── hooks/
│   │   │   └── useSocket.js
│   │   └── index.js
│   └── package.json
├── requirements.txt
└── README.md
```

## Setup & Run

### Python Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train PPO (recommended first — best model)
python train/train_ppo.py

# 4. (Optional) Train A2C
python train/train_a2c.py

# 5. (Optional) Train DQN
python train/train_dqn.py

# 6. Start inference server (after training)
python inference/server.py
```

### Frontend Setup

```bash
# In a new terminal
cd frontend
npm install
npm start
# Opens at http://localhost:3000
```

### Algorithm Comparison

After training all three models, run the comparison script:

```bash
python train/compare_algorithms.py
```

This evaluates each model on 20 random tracks and produces:
- A formatted comparison table in the terminal
- A bar chart saved to `comparison_results.png`

### Running Inference for a Specific Algorithm

The inference server reads the `MODEL_TYPE` environment variable (default: `PPO`).

```bash
# PPO (default)
python inference/server.py

# DQN — server automatically applies DiscreteActionWrapper
MODEL_TYPE=DQN python inference/server.py

# A2C
MODEL_TYPE=A2C python inference/server.py
```

The `/health` endpoint reports the active algorithm:

```json
{"status": "ok", "model": "DQN"}
```

### TensorBoard (optional)

```bash
tensorboard --logdir ./logs/
# Opens at http://localhost:6006
```

## Ablation Study

The three training scripts implement a deliberate ablation across RL algorithm families:

| Algorithm | Key Property | Expected Behavior |
|-----------|-------------|-------------------|
| **DQN** | Discrete actions (9 choices), replay buffer, target network | Struggles to find smooth racing lines due to quantized steering. Converges slowly. |
| **A2C** | Continuous actions, no trust-region clipping | Learns faster than DQN initially but may exhibit unstable policy updates (catastrophic forgetting on unseen tracks). |
| **PPO** | Continuous actions + `clip_range=0.2` | Stable monotonic improvement. Best zero-shot generalization to unseen tracks due to trust-region constraint. |

### Why PPO Wins

PPO's epsilon-clipping constraint (`clip_range=0.2`) prevents excessively large policy updates. This means:

1. **Stability** — The policy improves monotonically rather than oscillating.
2. **Generalisation** — Because updates are conservative, the agent doesn't overfit to specific track layouts.
3. **Smooth steering** — Continuous action space + stable updates → smooth racing lines.

## Reward Structure

| Component | Formula | Purpose |
|-----------|---------|---------|
| **Time penalty** | `-0.1` per step | Forces the agent to go fast — every wasted step costs reward |
| **Progress reward** | `+1.0 × Δprogress` | Rewards advancing along the track centerline |
| **Speed bonus** | `+0.05 × speed` | Small additional speed signal |
| **Smoothness penalty** | `-0.05 × \|Δsteer\|` | Discourages jerky steering |
| **Collision** | `-10` (terminal) | Crash ends the episode |

**Design philosophy:** No wall proximity penalty — in real racing (F1), optimal racing lines involve getting close to walls on corners (apex clipping). The agent learns its own risk tolerance.

## Expected Output

### Terminal (server)

```
INFO:     Uvicorn running on http://0.0.0.0:8000
Client connected: <socket_id>
```

### Browser (http://localhost:3000)

- **Left side:** 800×600 black canvas with glowing cyan track, neon car, and 7 color-coded LiDAR rays
- **Right side:** 320px dark panel with connection badge, speed gauge, lap progress, reward display, two scrolling graphs, and 11-element state vector

**Update rate:** ~30 frames per second

## Technical Details

### Environment

- **Track generation:** 30 random control points → Catmull-Rom spline → ±40px boundary walls
- **LiDAR:** 7 rays from -90° to +90° relative to heading, max range 200 units
- **Physics:** Kinematic bicycle model (wheelbase=30, max_speed=20, max_steer=0.5 rad)
- **Collision:** Point-in-polygon test via matplotlib.path.Path
- **Progress tracking:** Closest centerline point search with forward-only windowed matching

### Observation Space (11 floats)

| Index | Description | Range |
|-------|-------------|-------|
| 0–6 | 7 LiDAR distances (normalised) | [0, 1] |
| 7 | Speed (normalised) | [0, 1] |
| 8 | Steering angle (normalised) | [-1, 1] |
| 9 | Track progress fraction | [0, 1] |
| 10 | Heading alignment with track | [-1, 1] |
