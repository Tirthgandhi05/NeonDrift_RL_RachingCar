import React from "react";
import RaceCanvas from "./components/RaceCanvas";
import TelemetryPanel from "./components/TelemetryPanel";
import { useSocket } from "./hooks/useSocket";

/**
 * NeonDrift — Root Layout.
 *
 * Left:  800×600 RaceCanvas (car, track, LiDAR)
 * Right: 320px TelemetryPanel (speed gauge, reward, graphs, state vector)
 */
function App() {
  const { telemetry, connected } = useSocket();

  return (
    <div
      style={{
        display: "flex",
        width: "100vw",
        height: "100vh",
        background: "#000",
        fontFamily: "'Courier New', monospace",
        color: "#E0E0FF",
      }}
    >
      {/* Canvas area */}
      <div
        style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <RaceCanvas telemetry={telemetry} />
      </div>

      {/* Telemetry side panel */}
      <TelemetryPanel telemetry={telemetry} connected={connected} />
    </div>
  );
}

export default App;
