"""
NeonDrift — Custom Gymnasium Environment.

A 2D autonomous racing environment where an RL agent navigates a
procedurally-generated closed-loop track using simulated 1D LiDAR.
No computer vision, no pixels, no CNNs — the agent perceives the world
entirely through 7 ray-distance readings plus its own speed and steering.

Physics: Simple 2D kinematic bicycle model.
Track:   Catmull-Rom spline through 30 perturbed control points,
         offset ±TRACK_HALF_WIDTH for left/right boundary walls.

Observation (11 floats):
    [0..6]  7 normalised LiDAR distances               [0, 1]
    [7]     normalised longitudinal speed                [0, 1]
    [8]     normalised steering angle                    [-1, 1]
    [9]     track progress fraction                      [0, 1]
    [10]    heading alignment with track direction       [-1, 1]

Action (continuous, Box(2)):
    action[0]  steering delta   [-1, 1]
    action[1]  throttle/brake   [-1, 1]

Reward structure:
    -0.1          per step (time penalty — forces speed)
    +1.0 × Δprog  progress along centerline
    +0.05 × speed speed bonus
    -0.05 × |Δs|  smoothness penalty
    -50           collision (terminal)
"""

from __future__ import annotations

import gymnasium
import numpy as np
from gymnasium import spaces
from matplotlib.path import Path as MplPath
# scipy.signal.savgol_filter removed — was imported but never used

# ──────────────────────── Constants ────────────────────────
NUM_RAYS = 7
MAX_RAY_LEN = 200.0          # simulation units
MAX_SPEED = 20.0              # units per step (doubled for racing feel)
MAX_STEER = 0.5               # radians
TRACK_HALF_WIDTH = 40         # pixels / sim-units
CANVAS_W, CANVAS_H = 3000, 3000
PADDING = 150
WHEELBASE = 30.0              # distance between axles
MIN_TRACK_AREA = 50000        # reject self-intersecting tracks below this


# ──────────────────── Catmull-Rom Helpers ──────────────────
def catmull_rom_point(p0, p1, p2, p3, t):
    """Compute a single Catmull-Rom spline point."""
    return 0.5 * (
        2 * p1
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3
    )


def catmull_rom_chain(points: list, num_points: int = 20) -> list:
    """Return a smooth closed chain of points through the given control points."""
    pts = [points[-1]] + points + [points[0], points[1]]
    result = []
    for i in range(1, len(pts) - 2):
        for t_val in np.linspace(0, 1, num_points, endpoint=False):
            p = catmull_rom_point(pts[i - 1], pts[i], pts[i + 1], pts[i + 2], t_val)
            result.append(p)
    return result

def _ccw(A, B, C):
    """Helper for segment intersection check."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def _segments_intersect(A, B, C, D):
    """Return True if line segment AB intersects CD."""
    return _ccw(A, B, C) != _ccw(A, B, D) and _ccw(C, D, A) != _ccw(C, D, B)

def _is_self_intersecting(boundary: list) -> bool:
    """Check if a generated boundary curve self-intersects."""
    n = len(boundary)
    for i in range(n):
        a = boundary[i]
        b = boundary[(i + 1) % n]
        for j in range(i + 2, n):
            # Skip the last segment checking against the first segment
            if i == 0 and j == n - 1:
                continue
            c = boundary[j]
            d = boundary[(j + 1) % n]
            if _segments_intersect(a, b, c, d):
                return True
    return False


# ──────────────── Ray–Segment Intersection ─────────────────
def ray_segment_intersect(ray_origin, ray_end, seg_start, seg_end):
    """
    Returns distance from ray_origin to intersection with the segment,
    or None if no intersection.
    """
    d = ray_end - ray_origin
    f = seg_start - ray_origin
    b = seg_end - seg_start
    denom = d[0] * b[1] - d[1] * b[0]
    if abs(denom) < 1e-10:
        return None  # parallel
    t = (f[0] * b[1] - f[1] * b[0]) / denom
    u = (f[0] * d[1] - f[1] * d[0]) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return t * np.linalg.norm(d)
    return None


# ──────────────── Polygon Area (Shoelace) ──────────────────
def _polygon_area(pts):
    """Shoelace formula for signed polygon area."""
    pts = np.asarray(pts)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ═══════════════════════ Environment ═══════════════════════
class NeonDriftEnv(gymnasium.Env):
    """
    NeonDrift 2D racing environment.

    The track is procedurally regenerated every ``reset()`` so the agent
    cannot memorise a fixed layout.  Perception is 1D LiDAR only.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, max_steps: int = 2000):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        # Observation: 7 LiDAR + speed + steering + progress + heading_align = 11
        # Per-element bounds: LiDAR[0..6] in [0,1], speed in [0,1],
        # steer in [-1,1], progress in [0,1], heading_align in [-1,1]
        obs_low  = np.array([0]*7 + [0, -1, 0, -1], dtype=np.float32)
        obs_high = np.array([1]*7 + [1,  1, 1,  1], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        # Continuous action: [steer_delta, throttle]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # State variables (set in reset)
        self.car_x: float = 0.0
        self.car_y: float = 0.0
        self.car_heading: float = 0.0
        self.car_speed: float = 0.0
        self.car_steer: float = 0.0
        self.prev_steer: float = 0.0

        self.centerline: list = []
        self.left_boundary: list = []
        self.right_boundary: list = []
        self.track_polygon: MplPath | None = None
        self.boundary_segments: list = []
        self.lidar_readings = np.zeros(NUM_RAYS, dtype=np.float32)

        # Progress tracking along centerline
        self.progress_index: int = 0
        self.prev_progress_index: int = 0
        self.lap_completed: bool = False

        self.current_step = 0

        # Pygame surface (lazy init)
        self._screen = None
        self._clock = None

    # ─────────────────── Track Generation ──────────────────
    def _generate_track(self) -> None:
        """Generate a procedural closed-loop track via Catmull-Rom spline."""
        rng = self.np_random

        for _ in range(100):  # retry to avoid degenerate tracks
            # 30 control points on a perturbed circle
            n_ctrl = 30
            cx, cy = CANVAS_W / 2, CANVAS_H / 2
            base_radius = min(CANVAS_W, CANVAS_H) / 2 - PADDING - TRACK_HALF_WIDTH
            angles = np.linspace(0, 2 * np.pi, n_ctrl, endpoint=False)
            radii = base_radius + rng.uniform(-base_radius * 0.25, base_radius * 0.25, n_ctrl)
            ctrl_pts = [
                np.array([cx + r * np.cos(a), cy + r * np.sin(a)])
                for r, a in zip(radii, angles)
            ]

            # Smooth spline
            self.centerline = catmull_rom_chain(ctrl_pts, num_points=20)

            # Compute normals → left/right boundaries
            self.left_boundary = []
            self.right_boundary = []
            n = len(self.centerline)
            for i in range(n):
                p_prev = np.asarray(self.centerline[(i - 1) % n])
                p_next = np.asarray(self.centerline[(i + 1) % n])
                tangent = p_next - p_prev
                tangent /= (np.linalg.norm(tangent) + 1e-8)
                normal = np.array([-tangent[1], tangent[0]])
                cp = np.asarray(self.centerline[i])
                self.left_boundary.append((cp + TRACK_HALF_WIDTH * normal).tolist())
                self.right_boundary.append((cp - TRACK_HALF_WIDTH * normal).tolist())

            # Area sanity check (reject self-intersecting / tiny tracks)
            area = _polygon_area(self.left_boundary + list(reversed(self.right_boundary)))
            if area >= MIN_TRACK_AREA:
                # Extra check: strict self-intersection guard for high-curvature bows
                if not _is_self_intersecting(self.left_boundary) and not _is_self_intersecting(self.right_boundary):
                    break

        # Pre-build boundary segments for LiDAR raycasting
        self.boundary_segments = []
        for boundary in (self.left_boundary, self.right_boundary):
            for i in range(len(boundary)):
                s = np.asarray(boundary[i])
                e = np.asarray(boundary[(i + 1) % len(boundary)])
                self.boundary_segments.append((s, e))

        # Pre-compute vectorised segment arrays for fast LiDAR
        starts = np.array([s for s, _ in self.boundary_segments])  # (N, 2)
        ends = np.array([e for _, e in self.boundary_segments])    # (N, 2)
        self._seg_starts = starts
        self._seg_dirs = ends - starts  # (N, 2)

        # Convert centerline items to lists (for JSON serialisation)
        self.centerline = [
            c.tolist() if isinstance(c, np.ndarray) else list(c)
            for c in self.centerline
        ]
        self.left_boundary = [
            c if isinstance(c, list) else list(c) for c in self.left_boundary
        ]
        self.right_boundary = [
            c if isinstance(c, list) else list(c) for c in self.right_boundary
        ]

    # ──────────────────── Car Placement ────────────────────
    def _place_car_on_track(self) -> None:
        """Place car at start of centerline, heading toward second point."""
        p0 = np.asarray(self.centerline[0])
        p1 = np.asarray(self.centerline[1])
        self.car_x = float(p0[0])
        self.car_y = float(p0[1])
        self.car_heading = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        self.car_speed = 0.0
        self.car_steer = 0.0
        self.prev_steer = 0.0

    # ──────────────────── Polygon Build ────────────────────
    def _build_track_polygon(self) -> None:
        boundary_pts = self.left_boundary + list(reversed(self.right_boundary))
        self.track_polygon = MplPath(boundary_pts)

    # ──────────────────── Collision ─────────────────────────
    def _check_collision(self) -> bool:
        return not self.track_polygon.contains_point([self.car_x, self.car_y])

    # ──────────────────── LiDAR (Vectorised) ───────────────
    def _cast_lidar(self) -> None:
        """
        Cast NUM_RAYS rays using vectorised NumPy intersection.

        Instead of looping over ~1200 segments in Python, this checks
        ALL segments at once per ray via broadcasting.  ~5-10× faster.
        """
        car_pos = np.array([self.car_x, self.car_y])
        angles = np.linspace(-np.pi / 2, np.pi / 2, NUM_RAYS) + self.car_heading

        # F = seg_starts - car_pos, shape (N, 2)
        F = self._seg_starts - car_pos
        B = self._seg_dirs  # (N, 2)

        for i, ray_angle in enumerate(angles):
            # Ray direction vector (length = MAX_RAY_LEN)
            d = np.array([np.cos(ray_angle), np.sin(ray_angle)]) * MAX_RAY_LEN

            # Cross-product denominator: d.x * B.y - d.y * B.x
            denom = d[0] * B[:, 1] - d[1] * B[:, 0]

            # Mask parallel segments (avoid division by zero)
            parallel = np.abs(denom) < 1e-10
            safe_denom = np.where(parallel, 1.0, denom)

            # Ray parameter t and segment parameter u
            t = (F[:, 0] * B[:, 1] - F[:, 1] * B[:, 0]) / safe_denom
            u = (F[:, 0] * d[1] - F[:, 1] * d[0]) / safe_denom

            # Valid hits: not parallel, 0 <= t <= 1, 0 <= u <= 1
            valid = (~parallel) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)

            if np.any(valid):
                self.lidar_readings[i] = np.min(t[valid]) * MAX_RAY_LEN
            else:
                self.lidar_readings[i] = MAX_RAY_LEN

    # ──────────────────── Observation ──────────────────────
    def _get_obs(self) -> np.ndarray:
        normalised_lidar = self.lidar_readings / MAX_RAY_LEN
        speed_norm = self.car_speed / MAX_SPEED
        steer_norm = self.car_steer / MAX_STEER
        # Track progress as fraction [0, 1]
        n = len(self.centerline)
        progress_frac = self.progress_index / max(n, 1)
        # Heading alignment with track direction [-1, 1]
        heading_align = self._get_heading_alignment()
        obs = np.concatenate([
            normalised_lidar,
            [speed_norm, steer_norm, progress_frac, heading_align],
        ])
        return obs.astype(np.float32)

    # ──────────────── Heading Alignment ────────────────────
    def _get_heading_alignment(self) -> float:
        """Cosine similarity between car heading and local track direction."""
        n = len(self.centerline)
        if n < 2:
            return 0.0
        idx = self.progress_index % n
        p_curr = np.asarray(self.centerline[idx])
        p_next = np.asarray(self.centerline[(idx + 1) % n])
        track_dir = p_next - p_curr
        track_norm = np.linalg.norm(track_dir)
        if track_norm < 1e-8:
            return 0.0
        track_dir /= track_norm
        car_dir = np.array([np.cos(self.car_heading), np.sin(self.car_heading)])
        return float(np.clip(np.dot(car_dir, track_dir), -1.0, 1.0))

    # ──────────────── Progress Tracking ────────────────────
    def _update_progress(self) -> int:
        """
        Find the closest centerline point ahead of the current progress_index
        and return the number of new points passed (delta).
        """
        n = len(self.centerline)
        if n == 0:
            return 0
        car_pos = np.array([self.car_x, self.car_y])
        # Search in a window around current progress to avoid jumping
        best_idx = self.progress_index
        best_dist = float('inf')
        search_range = min(30, n // 2)  # look ahead up to 30 points
        for offset in range(search_range):
            idx = (self.progress_index + offset) % n
            pt = np.asarray(self.centerline[idx])
            dist = np.linalg.norm(car_pos - pt)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        self.prev_progress_index = self.progress_index
        # Calculate delta (handle wraparound)
        if best_idx >= self.progress_index:
            delta = best_idx - self.progress_index
        else:
            delta = (n - self.progress_index) + best_idx
        # Only accept forward progress (ignore tiny noise or backward jumps)
        if delta > 0 and delta < search_range:
            self.progress_index = best_idx
            # Lap completion: passed ≥ 95 % of centerline points and
            # the car is back near the start of the track
            if self.progress_index >= int(n * 0.95):
                start_pt = np.asarray(self.centerline[0])
                if np.linalg.norm(car_pos - start_pt) < TRACK_HALF_WIDTH * 2:
                    self.lap_completed = True
            return delta
        return 0

    # ──────────────────── Info Dict ────────────────────────
    def _get_info(self) -> dict:
        return {
            "speed": self.car_speed,
            "lidar": (self.lidar_readings / MAX_RAY_LEN).tolist(),
            "car_x": self.car_x,
            "car_y": self.car_y,
            "car_heading": float(self.car_heading),
            "left_boundary": self.left_boundary,
            "right_boundary": self.right_boundary,
            "centerline": self.centerline,
            "progress_pct": (self.progress_index / max(len(self.centerline), 1)) * 100,
        }

    # ═══════════════ Gymnasium API: reset ═══════════════════
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._generate_track()
        self._place_car_on_track()
        self._build_track_polygon()
        self._cast_lidar()
        self.current_step = 0
        self.progress_index = 0
        self.prev_progress_index = 0
        self.lap_completed = False
        return self._get_obs(), self._get_info()

    # ═══════════════ Gymnasium API: step ════════════════════
    def step(self, action):
        self.current_step += 1

        # ---- Apply actions ----
        steer_delta = float(action[0]) * MAX_STEER
        throttle = float(action[1])

        # Update steering (more responsive for high-speed racing)
        self.car_steer = np.clip(
            self.car_steer + steer_delta * 0.3, -MAX_STEER, MAX_STEER
        )
        # Update speed (faster acceleration for racing feel)
        self.car_speed += throttle * 1.0
        self.car_speed = float(np.clip(self.car_speed, 0.0, MAX_SPEED))

        # Update heading (bicycle model)
        if self.car_speed > 0.01:
            self.car_heading += (self.car_speed / WHEELBASE) * np.tan(self.car_steer)

        # Update position
        self.car_x += self.car_speed * np.cos(self.car_heading)
        self.car_y += self.car_speed * np.sin(self.car_heading)

        # ---- LiDAR ----
        self._cast_lidar()

        # ---- Progress tracking ----
        progress_delta = self._update_progress()

        # ---- Reward ----
        # 1. Time penalty: constant negative per step → forces speed
        time_penalty = -0.1

        # 2. Progress reward: advance along the centerline
        progress_reward = 1.0 * progress_delta

        # 3. Speed bonus: small additional signal
        speed_bonus = 0.05 * self.car_speed

        # 4. Smoothness penalty: discourage jerky steering
        steer_change = abs(self.car_steer - self.prev_steer)
        smooth_penalty = -0.05 * steer_change

        if self._check_collision():
            reward = -10.0
            terminated = True
        elif self.lap_completed:
            reward = time_penalty + progress_reward + speed_bonus + smooth_penalty + 100.0
            terminated = True
        else:
            reward = time_penalty + progress_reward + speed_bonus + smooth_penalty
            terminated = False

        self.prev_steer = self.car_steer

        # ---- Truncation ----
        truncated = self.current_step >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    # ═══════════════ Gymnasium API: render ══════════════════
    def render(self):
        """Render the environment using pygame (for local debugging only)."""
        try:
            import pygame
        except ImportError:
            return None

        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((CANVAS_W, CANVAS_H))
                pygame.display.set_caption("NeonDrift")
            else:
                self._screen = pygame.Surface((CANVAS_W, CANVAS_H))
            self._clock = pygame.time.Clock()

        surf = self._screen
        surf.fill((0, 0, 0))

        # Draw boundaries
        if len(self.left_boundary) > 2:
            pygame.draw.lines(surf, (0, 255, 255), True,
                              [(int(p[0]), int(p[1])) for p in self.left_boundary], 2)
            pygame.draw.lines(surf, (0, 255, 255), True,
                              [(int(p[0]), int(p[1])) for p in self.right_boundary], 2)

        # Draw centerline
        if len(self.centerline) > 2:
            pygame.draw.lines(surf, (255, 255, 255), True,
                              [(int(p[0]), int(p[1])) for p in self.centerline], 1)

        # Draw LiDAR rays
        car_pos = np.array([self.car_x, self.car_y])
        angles = np.linspace(-np.pi / 2, np.pi / 2, NUM_RAYS)
        for i, angle_offset in enumerate(angles):
            ray_angle = self.car_heading + angle_offset
            hit_dist = self.lidar_readings[i]
            end = car_pos + hit_dist * np.array([np.cos(ray_angle), np.sin(ray_angle)])
            frac = hit_dist / MAX_RAY_LEN
            if frac > 0.5:
                color = (0, 255, 0)
            elif frac > 0.3:
                color = (255, 255, 0)
            else:
                color = (255, 0, 0)
            pygame.draw.line(surf, color,
                             (int(self.car_x), int(self.car_y)),
                             (int(end[0]), int(end[1])), 1)

        # Draw car (triangle)
        length, width = 18, 10
        heading = self.car_heading
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        nose = (self.car_x + length * cos_h, self.car_y + length * sin_h)
        rl = (self.car_x - length / 2 * cos_h + width / 2 * (-sin_h),
              self.car_y - length / 2 * sin_h + width / 2 * cos_h)
        rr = (self.car_x - length / 2 * cos_h - width / 2 * (-sin_h),
              self.car_y - length / 2 * sin_h - width / 2 * cos_h)
        pygame.draw.polygon(surf, (255, 0, 170),
                            [(int(p[0]), int(p[1])) for p in [nose, rl, rr]])

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2)
            )

    def close(self):
        if self._screen is not None:
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
            self._screen = None


# ═══════════════ Discrete Action Wrapper (DQN) ═════════════
class DiscreteActionWrapper(gymnasium.ActionWrapper):
    """
    Maps 9 discrete action indices to continuous [steer_delta, throttle]
    bundles.  Used exclusively for DQN training — the base env is
    always continuous.

    Actions:
        0: Hard Left + Gas      [-1.0,  1.0]
        1: Soft Left + Gas      [-0.5,  1.0]
        2: Slight Left + Gas    [-0.25, 1.0]
        3: Straight  + Gas      [ 0.0,  1.0]
        4: Slight Right + Gas   [ 0.25, 1.0]
        5: Soft Right + Gas     [ 0.5,  1.0]
        6: Hard Right + Gas     [ 1.0,  1.0]
        7: Straight  + Brake    [ 0.0, -0.5]
        8: Straight  + Coast    [ 0.0,  0.3]
    """

    ACTIONS = [
        np.array([-1.0,  1.0], dtype=np.float32),  # 0: Hard Left + Gas
        np.array([-0.5,  1.0], dtype=np.float32),  # 1: Soft Left + Gas
        np.array([-0.25, 1.0], dtype=np.float32),  # 2: Slight Left + Gas
        np.array([ 0.0,  1.0], dtype=np.float32),  # 3: Straight + Gas
        np.array([ 0.25, 1.0], dtype=np.float32),  # 4: Slight Right + Gas
        np.array([ 0.5,  1.0], dtype=np.float32),  # 5: Soft Right + Gas
        np.array([ 1.0,  1.0], dtype=np.float32),  # 6: Hard Right + Gas
        np.array([ 0.0, -0.5], dtype=np.float32),  # 7: Straight + Brake
        np.array([ 0.0,  0.3], dtype=np.float32),  # 8: Straight + Coast
    ]

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gymnasium.spaces.Discrete(len(self.ACTIONS))

    def action(self, action_idx):
        return self.ACTIONS[action_idx]