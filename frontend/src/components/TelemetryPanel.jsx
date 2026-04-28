import React, { useState, useEffect } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";

/**
 * TelemetryPanel — Right-side 320px panel.
 *
 * Sections:
 *   1. Connection status badge
 *   2. Speed gauge (horizontal bar)
 *   3. Reward display (green/red)
 *   4. Two live graphs (speed & reward over time)
 *   5. State vector display (9 floats, color-coded)
 *
 * Props:
 *   telemetry : object | null
 *   connected : boolean
 */

const MAX_SPEED = 20;
const MAX_HISTORY = 100;

const panelStyle = {
  width: 320,
  minWidth: 320,
  height: "100vh",
  background: "rgba(10, 10, 18, 0.65)",
  backdropFilter: "blur(12px)",
  WebkitBackdropFilter: "blur(12px)",
  padding: "20px 16px",
  overflowY: "auto",
  fontFamily: "'Inter', 'Outfit', 'Segoe UI', sans-serif",
  color: "#E0E0FF",
  borderLeft: "1px solid rgba(0, 255, 255, 0.15)",
  boxShadow: "-5px 0 25px rgba(0, 0, 0, 0.5)",
  display: "flex",
  flexDirection: "column",
  gap: 20,
  zIndex: 10,
};

const headerStyle = {
  fontSize: 14,
  fontWeight: "bold",
  letterSpacing: 2,
  textTransform: "uppercase",
  textShadow: "0 0 8px #00FFFF",
  color: "#00FFFF",
  marginBottom: 6,
};

const labelStyle = {
  fontSize: 11,
  letterSpacing: 1,
  textTransform: "uppercase",
  color: "#888",
  marginBottom: 4,
};

const valueStyle = {
  fontSize: 18,
  fontWeight: "bold",
};

function TelemetryPanel({ telemetry, connected }) {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    if (!telemetry) return;
    setHistory((prev) => {
      const next = [
        ...prev,
        {
          t: prev.length,
          speed: telemetry.speed,
          reward: telemetry.reward,
        },
      ];
      return next.slice(-MAX_HISTORY);
    });
  }, [telemetry]);

  const speed = telemetry ? telemetry.speed : 0;
  const reward = telemetry ? telemetry.reward : 0;
  const stateVector = telemetry ? telemetry.state_vector : null;
  const progressPct = telemetry ? (telemetry.progress_pct || 0) : 0;
  const speedBarWidth = Math.min((speed / MAX_SPEED) * 100, 100);

  return (
    <div style={panelStyle}>
      {/* ── Title ── */}
      <div
        style={{
          fontSize: 18,
          fontWeight: "bold",
          textAlign: "center",
          textShadow: "0 0 12px #FF00AA",
          color: "#FF00AA",
          letterSpacing: 3,
        }}
      >
        NEONDRIFT
      </div>

      {/* ── 1. Connection Status ── */}
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span
          style={{
            display: "inline-block",
            width: 10,
            height: 10,
            borderRadius: "50%",
            background: connected ? "#00FF00" : "#FF0000",
            boxShadow: connected
              ? "0 0 8px #00FF00"
              : "0 0 8px #FF0000",
          }}
        />
        <span
          style={{
            fontSize: 12,
            letterSpacing: 2,
            color: connected ? "#00FF00" : "#FF0000",
          }}
        >
          {connected ? "CONNECTED" : "DISCONNECTED"}
        </span>
      </div>

      {/* ── 2. Speed Gauge ── */}
      <div>
        <div style={labelStyle}>SPEED</div>
        <div style={valueStyle}>
          {speed.toFixed(2)} <span style={{ fontSize: 11, color: "#888" }}>u/s</span>
        </div>
        <div
          style={{
            marginTop: 6,
            height: 8,
            background: "#1a1a2e",
            borderRadius: 4,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              width: `${speedBarWidth}%`,
              height: "100%",
              background: "linear-gradient(90deg, #006666, #00FFFF)",
              boxShadow: "0 0 8px #00FFFF",
              borderRadius: 4,
              transition: "width 0.05s linear",
            }}
          />
        </div>
      </div>

      {/* ── 3. Reward Display ── */}
      <div>
        <div style={labelStyle}>REWARD</div>
        <div
          style={{
            ...valueStyle,
            color: reward >= 0 ? "#00FF00" : "#FF0000",
            textShadow: reward >= 0
              ? "0 0 6px #00FF00"
              : "0 0 6px #FF0000",
          }}
        >
          {reward.toFixed(3)}
        </div>
      </div>

      {/* ── 3b. Lap Progress ── */}
      <div>
        <div style={labelStyle}>LAP PROGRESS</div>
        <div
          style={{
            ...valueStyle,
            color: "#FF00AA",
            textShadow: "0 0 6px #FF00AA",
          }}
        >
          {progressPct.toFixed(1)}%
        </div>
        <div
          style={{
            marginTop: 6,
            height: 8,
            background: "#1a1a2e",
            borderRadius: 4,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              width: `${Math.min(progressPct, 100)}%`,
              height: "100%",
              background: "linear-gradient(90deg, #660044, #FF00AA)",
              boxShadow: "0 0 8px #FF00AA",
              borderRadius: 4,
              transition: "width 0.05s linear",
            }}
          />
        </div>
      </div>

      {/* ── 4. Live Graphs ── */}
      <div>
        <div style={labelStyle}>SPEED OVER TIME</div>
        <div style={{ width: "100%", height: 100 }}>
          <ResponsiveContainer width="100%" height="100%">
            {/* Switched to AreaChart with gradient to look vastly better */}
            <AreaChart data={history}>
              <defs>
                <linearGradient id="colorSpeed" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#00FFFF" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#00FFFF" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(0, 255, 255, 0.1)" strokeDasharray="3 3"/>
              <XAxis dataKey="t" hide />
              <YAxis domain={[0, MAX_SPEED]} hide />
              <Area
                type="monotone"
                dataKey="speed"
                stroke="#00FFFF"
                fillOpacity={1}
                fill="url(#colorSpeed)"
                dot={false}
                strokeWidth={2}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div>
        <div style={labelStyle}>REWARD OVER TIME</div>
        <div style={{ width: "100%", height: 100 }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={history}>
              <defs>
                <linearGradient id="colorReward" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#FF00AA" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#FF00AA" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(255, 0, 170, 0.1)" strokeDasharray="3 3"/>
              <XAxis dataKey="t" hide />
              <YAxis hide />
              <Area
                type="monotone"
                dataKey="reward"
                stroke="#FF00AA"
                fillOpacity={1}
                fill="url(#colorReward)"
                dot={false}
                strokeWidth={2}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── 5. State Vector Display ── */}
      <div>
        <div style={labelStyle}>STATE VECTOR [9]</div>
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "4px 8px",
            fontSize: 13,
            fontFamily: "'Courier New', monospace",
          }}
        >
          {stateVector ? (
            <>
              <span style={{ color: "#888" }}>[</span>
              {stateVector.map((val, i) => {
                let color = "#E0E0FF";
                if (val < 0.3) color = "#FF0000";
                else if (val > 0.7) color = "#00FFFF";
                return (
                  <span key={i} style={{ color }}>
                    {val >= 0 ? " " : ""}
                    {val.toFixed(3)}
                  </span>
                );
              })}
              <span style={{ color: "#888" }}>]</span>
            </>
          ) : (
            <span style={{ color: "#555" }}>[ — ]</span>
          )}
        </div>
      </div>
    </div>
  );
}

export default TelemetryPanel;
