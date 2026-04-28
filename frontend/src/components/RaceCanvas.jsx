import React, { useRef, useEffect } from "react";

/**
 * RaceCanvas — 800×600 HTML5 canvas renderer.
 *
 * Drawing order (back to front):
 *   1. Clear canvas (black)
 *   2. Right boundary (cyan, glow)
 *   3. Left boundary (cyan, glow)
 *   4. Centerline (white, dashed, low opacity)
 *   5. LiDAR rays (color-coded, opacity 0.6)
 *   6. Car polygon (neon pink triangle)
 *
 * Props:
 *   telemetry : object | null — latest telemetry from server
 */

const CANVAS_W = 800;
const CANVAS_H = 600;

// ──────────────── Drawing Helpers ─────────────────────────

function drawGrid(ctx, cx, cy) {
  ctx.save();
  ctx.strokeStyle = "rgba(255, 0, 170, 0.15)";
  ctx.lineWidth = 1;
  const gridSize = 50;
  const offsetX = -(cx % gridSize);
  const offsetY = -(cy % gridSize);

  ctx.beginPath();
  for (let x = offsetX - gridSize; x <= CANVAS_W + gridSize; x += gridSize) {
    ctx.moveTo(x, 0);
    ctx.lineTo(x, CANVAS_H);
  }
  for (let y = offsetY - gridSize; y <= CANVAS_H + gridSize; y += gridSize) {
    ctx.moveTo(0, y);
    ctx.lineTo(CANVAS_W, y);
  }
  ctx.stroke();
  ctx.restore();
}

function drawAsphalt(ctx, left, right) {
  if (!left || !right || left.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(left[0][0], left[0][1]);
  for (let i = 1; i < left.length; i++) {
    ctx.lineTo(left[i][0], left[i][1]);
  }
  for (let i = right.length - 1; i >= 0; i--) {
    ctx.lineTo(right[i][0], right[i][1]);
  }
  ctx.closePath();
  ctx.fillStyle = "#080812"; // very dark synthwave asphalt
  ctx.fill();
}

function drawPolyline(ctx, points, color) {
  if (!points || points.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i][0], points[i][1]);
  }
  ctx.closePath(); // boundary is a closed loop
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.shadowBlur = 10;
  ctx.shadowColor = color;
  ctx.stroke();
}

function drawCenterline(ctx, points) {
  if (!points || points.length < 2) return;
  ctx.save();
  ctx.globalAlpha = 0.15;
  ctx.setLineDash([5, 10]);
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i][0], points[i][1]);
  }
  ctx.closePath();
  ctx.strokeStyle = "#FFFFFF";
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
}

function drawLidarRay(ctx, ray) {
  const { x1, y1, x2, y2, hit_fraction } = ray;
  let color;
  if (hit_fraction > 0.5) color = "#00FF00";
  else if (hit_fraction > 0.3) color = "#FFFF00";
  else color = "#FF0000";

  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.globalAlpha = 0.6;
  ctx.shadowBlur = 5;
  ctx.shadowColor = color;
  ctx.stroke();
  ctx.globalAlpha = 1.0;
}

function drawCar(ctx, x, y, heading, speed) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(heading);

  // Drop shadow
  ctx.shadowColor = "rgba(0, 255, 255, 0.5)";
  ctx.shadowBlur = 15;
  ctx.shadowOffsetX = -2;
  ctx.shadowOffsetY = 2;

  // Car Body
  ctx.fillStyle = "#111115"; 
  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(-12, -7, 26, 14, 4);
  } else {
    ctx.fillRect(-12, -7, 26, 14);
  }
  ctx.fill();
  
  // Neon Side Skirts
  ctx.strokeStyle = "#00FFFF";
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.shadowColor = "transparent";

  // Glass Cockpit
  ctx.fillStyle = "#0A0A0F";
  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(-4, -5, 12, 10, 2);
  } else {
    ctx.fillRect(-4, -5, 12, 10);
  }
  ctx.fill();
  ctx.strokeStyle = "#FF00AA";
  ctx.lineWidth = 1;
  ctx.stroke();

  // Headlights (glowing cyan)
  ctx.fillStyle = "#00FFFF";
  ctx.shadowColor = "#00FFFF";
  ctx.shadowBlur = 10;
  ctx.fillRect(12, -6, 2, 3);
  ctx.fillRect(12, 3, 2, 3);
  ctx.shadowColor = "transparent";
  
  // Exhaust flame (scales with speed)
  if (speed > 0.5) {
    const flameLen = (speed / 10) * 15 + Math.random() * 6;
    ctx.fillStyle = "#FF00AA";
    ctx.shadowColor = "#FF00AA";
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.moveTo(-13, -3);
    ctx.lineTo(-13 - flameLen, 0);
    ctx.lineTo(-13, 3);
    ctx.fill();
  }

  ctx.restore();
}

// ──────────────── Component ──────────────────────────────

function RaceCanvas({ telemetry }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d");

    // 1. Clear
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

    if (!telemetry) {
      // Draw waiting message
      ctx.fillStyle = "#00FFFF";
      ctx.font = "20px 'Courier New', monospace";
      ctx.textAlign = "center";
      ctx.shadowBlur = 15;
      ctx.shadowColor = "#00FFFF";
      ctx.fillText("Waiting for telemetry...", CANVAS_W / 2, CANVAS_H / 2);
      ctx.shadowBlur = 0;
      return;
    }

    // --- Synthetic Moving Camera ---
    const cx = CANVAS_W / 2 - telemetry.car_x;
    const cy = CANVAS_H / 2 - telemetry.car_y;

    // 2. Draw static shifting grid (background)
    drawGrid(ctx, telemetry.car_x, telemetry.car_y);

    ctx.save();
    // Move the entire world so the car is exactly at the center of the canvas
    ctx.translate(cx, cy);

    // 3. Draw Asphalt (filled polygon between inner and outer bounds)
    drawAsphalt(ctx, telemetry.left_boundary, telemetry.right_boundary);

    // 4. Right boundary (cyan glow)
    drawPolyline(ctx, telemetry.right_boundary, "#00FFFF");

    // 5. Left boundary (cyan glow)
    drawPolyline(ctx, telemetry.left_boundary, "#00FFFF");

    // 6. Centerline (white dashed, low opacity)
    drawCenterline(ctx, telemetry.centerline);

    // 7. LiDAR rays
    ctx.shadowBlur = 0;
    if (telemetry.lidar_rays) {
      telemetry.lidar_rays.forEach((ray) => drawLidarRay(ctx, ray));
    }

    // 8. Car
    ctx.shadowBlur = 0;
    drawCar(ctx, telemetry.car_x, telemetry.car_y, telemetry.car_heading, telemetry.speed);

    ctx.restore();

    // Reset shadow
    ctx.shadowBlur = 0;
    ctx.shadowColor = "transparent";
  }, [telemetry]);

  return (
    <canvas
      ref={canvasRef}
      width={CANVAS_W}
      height={CANVAS_H}
      style={{
        border: "1px solid #111",
        borderRadius: "4px",
        boxShadow: "0 0 30px rgba(0, 255, 255, 0.15)",
      }}
    />
  );
}

export default RaceCanvas;
