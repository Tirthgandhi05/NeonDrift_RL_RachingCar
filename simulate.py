"""
NeonDrift — Pygame Simulation Script.

Runs the trained DQN agent in a properly sized windowed display with
auto-zoom so the car and track always fit on screen.

Controls:
    R          — reset to a new track immediately
    +/-        — zoom in / out manually
    SPACE      — pause / unpause
    Q / ESC    — quit

Usage:
    python simulate.py                        # DQN (default)
    python simulate.py --model ppo_final      # PPO
    python simulate.py --episodes 20          # run 20 episodes
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pygame
from stable_baselines3 import DQN, PPO, A2C

from env.neondrift_env import NeonDriftEnv, DiscreteActionWrapper, MAX_RAY_LEN, NUM_RAYS

# ──────────────────────── Config ───────────────────────────
WINDOW_W, WINDOW_H = 1024, 768        # windowed — not fullscreen
FPS = 30
BG_COLOR        = (8,   8,  20)
TRACK_COLOR     = (0,  200, 220)
CENTER_COLOR    = (60,  60,  80)
CAR_COLOR       = (255,  0, 170)
LIDAR_COLORS    = [(255, 0, 0), (255, 165, 0), (0, 255, 0)]   # red/orange/green by dist
HUD_COLOR       = (200, 200, 220)
WARN_COLOR      = (255,  80,  80)
GOOD_COLOR      = (80,  255, 120)

# ──────────────────────── Args ─────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    default="dqn_final",  help="Model filename (without .zip) in ./models/")
    p.add_argument("--algo",     default="DQN",        help="Algorithm class: DQN, PPO, A2C")
    p.add_argument("--episodes", default=50, type=int, help="Number of episodes to run")
    p.add_argument("--fps",      default=30,  type=int, help="Simulation FPS")
    return p.parse_args()

# ──────────────────────── Load Model ───────────────────────
def load_model(algo: str, model_name: str):
    cls = {"DQN": DQN, "PPO": PPO, "A2C": A2C}[algo.upper()]
    paths = [
        f"./models/{model_name}",
        f"./models/{algo.lower()}_best/best_model",
        f"./models/{algo.lower()}_final",
    ]
    for path in paths:
        zip_path = path if path.endswith(".zip") else path + ".zip"
        if os.path.isfile(zip_path) or os.path.isfile(path):
            print(f"[simulate] Loading {algo} from: {path}")
            return cls.load(path)
    raise FileNotFoundError(f"No model found. Tried: {paths}")

# ──────────────────────── Camera ───────────────────────────
class Camera:
    """Tracks the car and auto-fits the track into the viewport."""

    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.zoom = 0.25        # start zoomed out to show whole track
        self.cx = w / 2
        self.cy = h / 2
        self.target_x = w / 2
        self.target_y = h / 2
        self.smooth = 0.08      # camera lag (lower = more lag)

    def fit_track(self, centerline):
        """Auto-zoom to fit the entire track on screen with padding."""
        if not centerline:
            return
        pts = np.array(centerline)
        min_x, min_y = pts.min(axis=0)
        max_x, max_y = pts.max(axis=0)
        track_w = max_x - min_x
        track_h = max_y - min_y
        padding = 100
        zoom_x = (self.w - padding * 2) / max(track_w, 1)
        zoom_y = (self.h - padding * 2) / max(track_h, 1)
        self.zoom = min(zoom_x, zoom_y)
        # Centre on track midpoint
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2
        self.cx = self.w / 2 - mid_x * self.zoom
        self.cy = self.h / 2 - mid_y * self.zoom
        self.target_x = self.cx
        self.target_y = self.cy

    def follow(self, car_x, car_y):
        """Smoothly pan so the car stays near the centre."""
        target_cx = self.w / 2 - car_x * self.zoom
        target_cy = self.h / 2 - car_y * self.zoom
        self.target_x = target_cx
        self.target_y = target_cy
        self.cx += (self.target_x - self.cx) * self.smooth
        self.cy += (self.target_y - self.cy) * self.smooth

    def world_to_screen(self, x, y):
        return (int(x * self.zoom + self.cx),
                int(y * self.zoom + self.cy))

    def adjust_zoom(self, delta):
        self.zoom = max(0.05, min(2.0, self.zoom + delta))

# ──────────────────────── Drawing ──────────────────────────
def draw_polyline(surf, color, pts, cam, closed=False, width=2):
    if len(pts) < 2:
        return
    screen_pts = [cam.world_to_screen(p[0], p[1]) for p in pts]
    pygame.draw.lines(surf, color, closed, screen_pts, width)


def draw_car(surf, cam, car_x, car_y, heading, speed):
    """Draw car as a pointed triangle, colour shifts with speed."""
    length = max(12, 18 * cam.zoom / 0.25)
    width  = max(7,  10 * cam.zoom / 0.25)
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    nose = (car_x + length * cos_h,      car_y + length * sin_h)
    rl   = (car_x - length/2 * cos_h + width/2 * (-sin_h),
            car_y - length/2 * sin_h + width/2 * cos_h)
    rr   = (car_x - length/2 * cos_h - width/2 * (-sin_h),
            car_y - length/2 * sin_h - width/2 * cos_h)
    pts = [cam.world_to_screen(*p) for p in [nose, rl, rr]]
    # Colour: cool pink → hot white with speed
    t = min(speed / 20.0, 1.0)
    r = int(255)
    g = int(t * 180)
    b = int(170 * (1 - t) + 255 * t)
    pygame.draw.polygon(surf, (r, g, b), pts)
    pygame.draw.polygon(surf, (255, 255, 255), pts, 1)


def draw_lidar(surf, cam, car_x, car_y, heading, lidar_readings):
    angles = np.linspace(-np.pi / 2, np.pi / 2, NUM_RAYS)
    for i, angle_offset in enumerate(angles):
        ray_angle = heading + angle_offset
        dist = lidar_readings[i]
        end_x = car_x + dist * np.cos(ray_angle)
        end_y = car_y + dist * np.sin(ray_angle)
        frac = dist / MAX_RAY_LEN
        if frac > 0.5:
            color = (0, 200, 80)
        elif frac > 0.25:
            color = (255, 200, 0)
        else:
            color = (255, 50, 50)
        sx, sy = cam.world_to_screen(car_x, car_y)
        ex, ey = cam.world_to_screen(end_x, end_y)
        pygame.draw.line(surf, color, (sx, sy), (ex, ey), 1)
        pygame.draw.circle(surf, color, (ex, ey), 3)


def draw_hud(surf, font, small_font, episode, total_episodes,
             reward, progress, speed, steps, paused, algo):
    """Draw telemetry overlay in top-left corner."""
    pad = 14
    line_h = 22

    lines = [
        (f"[ {algo} ]",                    (180, 180, 255)),
        (f"Episode  {episode}/{total_episodes}", HUD_COLOR),
        (f"Steps    {steps}",               HUD_COLOR),
        (f"Reward   {reward:+.1f}",         GOOD_COLOR if reward > 0 else WARN_COLOR),
        (f"Progress {progress:.1f}%",       GOOD_COLOR if progress > 50 else HUD_COLOR),
        (f"Speed    {speed:.1f}",           HUD_COLOR),
    ]
    if paused:
        lines.append(("  PAUSED  [SPACE]", (255, 220, 0)))

    # Background panel
    panel_w = 200
    panel_h = len(lines) * line_h + pad * 2
    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 160))
    surf.blit(panel, (8, 8))

    for i, (text, color) in enumerate(lines):
        rendered = small_font.render(text, True, color)
        surf.blit(rendered, (pad + 8, pad + 8 + i * line_h))

    # Controls hint at bottom
    hint = small_font.render("R=reset  +/-=zoom  SPACE=pause  Q=quit", True, (80, 80, 100))
    surf.blit(hint, (pad, WINDOW_H - 24))


def draw_episode_end(surf, font, crashed, total_reward, progress):
    """Flash a result banner briefly."""
    msg   = "💥 CRASHED" if crashed else "✅ COMPLETED"
    color = WARN_COLOR if crashed else GOOD_COLOR
    text  = font.render(f"{msg}  |  Reward: {total_reward:.0f}  |  Progress: {progress:.1f}%", True, color)
    rect  = text.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2))
    bg    = pygame.Surface((rect.width + 40, rect.height + 20), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 200))
    surf.blit(bg, (rect.x - 20, rect.y - 10))
    surf.blit(text, rect)

# ──────────────────────── Main ─────────────────────────────
def main():
    args = parse_args()

    # Load model
    model = load_model(args.algo, args.model)
    is_discrete = args.algo.upper() == "DQN"

    # Pygame setup — WINDOWED, not fullscreen
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))   # no FULLSCREEN flag
    pygame.display.set_caption(f"NeonDrift — {args.algo} Agent")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("consolas", 20, bold=True)
    small  = pygame.font.SysFont("consolas", 15)

    cam = Camera(WINDOW_W, WINDOW_H)

    # Stats across episodes
    all_rewards   = []
    all_progresses = []
    all_steps     = []

    episode = 0
    paused  = False

    while episode < args.episodes:
        # Create env
        base_env = NeonDriftEnv()
        env = DiscreteActionWrapper(base_env) if is_discrete else base_env
        obs, info = env.reset()

        # Fit camera to this track
        cam.fit_track(info["centerline"])

        total_reward = 0.0
        steps        = 0
        done         = False
        crashed      = False
        show_end     = False
        end_timer    = 0

        while True:
            # ── Events ──────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        pygame.quit(); sys.exit()
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    if event.key == pygame.K_r:
                        done = True          # force reset
                    if event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                        cam.adjust_zoom(+0.02)
                    if event.key == pygame.K_MINUS:
                        cam.adjust_zoom(-0.02)

            # ── Show end banner then move on ─────────────────
            if show_end:
                screen.fill(BG_COLOR)
                draw_polyline(screen, CENTER_COLOR, info["centerline"], cam, closed=True, width=1)
                draw_polyline(screen, TRACK_COLOR,  info["left_boundary"],  cam, closed=True, width=2)
                draw_polyline(screen, TRACK_COLOR,  info["right_boundary"], cam, closed=True, width=2)
                draw_episode_end(screen, font, crashed, total_reward, info.get("progress_pct", 0))
                pygame.display.flip()
                clock.tick(FPS)
                end_timer += 1
                if end_timer > FPS * 2:   # show for 2 seconds
                    break
                continue

            if done:
                break

            # ── Simulation step ──────────────────────────────
            if not paused:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps        += 1
                done          = terminated or truncated
                crashed       = terminated and not truncated

                if done:
                    all_rewards.append(total_reward)
                    all_progresses.append(info.get("progress_pct", 0))
                    all_steps.append(steps)
                    show_end  = True
                    end_timer = 0
                    continue

                # Smooth camera follow
                cam.follow(info["car_x"], info["car_y"])

            # ── Draw ─────────────────────────────────────────
            screen.fill(BG_COLOR)

            # Track
            draw_polyline(screen, CENTER_COLOR, info["centerline"],    cam, closed=True, width=1)
            draw_polyline(screen, TRACK_COLOR,  info["left_boundary"], cam, closed=True, width=2)
            draw_polyline(screen, TRACK_COLOR,  info["right_boundary"],cam, closed=True, width=2)

            # LiDAR + car
            lidar_raw = np.array(info["lidar"]) * MAX_RAY_LEN
            draw_lidar(screen, cam, info["car_x"], info["car_y"],
                       info["car_heading"], lidar_raw)
            draw_car(screen, cam, info["car_x"], info["car_y"],
                     info["car_heading"], info["speed"])

            # HUD
            draw_hud(screen, font, small,
                     episode + 1, args.episodes,
                     total_reward, info.get("progress_pct", 0),
                     info["speed"], steps, paused, args.algo)

            pygame.display.flip()
            clock.tick(args.fps)

        env.close()
        episode += 1

    # ── Final summary ────────────────────────────────────────
    screen.fill(BG_COLOR)
    lines = [
        f"  {args.algo} — {args.episodes} Episodes Complete",
        f"  Avg Reward   : {np.mean(all_rewards):.1f}",
        f"  Avg Progress : {np.mean(all_progresses):.1f}%",
        f"  Avg Steps    : {np.mean(all_steps):.0f}",
        f"  Crash Rate   : {sum(1 for r in all_rewards if r < -10) / len(all_rewards) * 100:.0f}%",
        "",
        "  Press Q to exit",
    ]
    for i, line in enumerate(lines):
        surf = font.render(line, True, HUD_COLOR)
        screen.blit(surf, (WINDOW_W // 2 - 200, WINDOW_H // 2 - 80 + i * 30))
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    waiting = False
        clock.tick(15)

    pygame.quit()


if __name__ == "__main__":
    main()