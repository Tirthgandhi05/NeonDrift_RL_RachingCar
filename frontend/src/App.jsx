import React, { useState } from "react";
import RaceCanvas from "./components/RaceCanvas";
import TelemetryPanel from "./components/TelemetryPanel";
import { useSocket } from "./hooks/useSocket";

/**
 * NeonDrift — Root Layout.
 *
 * Implements a dynamic selection screen to choose the RL algorithm,
 * then transitions to the simulation view.
 */
function App() {
  const { telemetry, connected, changeModel } = useSocket();
  const [activeScreen, setActiveScreen] = useState("selection");

  const handleSelectModel = (modelName) => {
    changeModel(modelName);
    setActiveScreen("simulation");
  };

  if (activeScreen === "selection") {
    return (
      <div style={{
        display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
        width: "100vw", height: "100vh", background: "#000", color: "#00FFFF", fontFamily: "'Courier New', monospace"
      }}>
        <h1 style={{ fontSize: "4rem", marginBottom: "1rem", textShadow: "0 0 20px #00FFFF, 0 0 40px #00FFFF", letterSpacing: "4px" }}>NEON DRIFT</h1>
        <p style={{ fontSize: "1.2rem", marginBottom: "4rem", color: "#E0E0FF", opacity: 0.8 }}>SELECT INFERENCE MODEL</p>
        
        <div style={{ display: "flex", gap: "2rem" }}>
          {["PPO", "A2C", "DQN"].map((model) => (
            <button 
              key={model}
              onClick={() => handleSelectModel(model)}
              style={{
                background: "rgba(0, 20, 40, 0.8)", border: "2px solid #00FFFF", color: "#00FFFF",
                padding: "1.5rem 4rem", fontSize: "1.5rem", cursor: "pointer", fontWeight: "bold",
                boxShadow: "0 0 15px rgba(0, 255, 255, 0.2), inset 0 0 10px rgba(0, 255, 255, 0.1)", 
                transition: "all 0.2s"
              }}
              onMouseOver={(e) => { 
                e.target.style.background = "rgba(0, 255, 255, 0.2)";
                e.target.style.boxShadow = "0 0 25px rgba(0, 255, 255, 0.5), inset 0 0 20px rgba(0, 255, 255, 0.3)";
              }}
              onMouseOut={(e) => { 
                e.target.style.background = "rgba(0, 20, 40, 0.8)";
                e.target.style.boxShadow = "0 0 15px rgba(0, 255, 255, 0.2), inset 0 0 10px rgba(0, 255, 255, 0.1)";
              }}
            >
              {model}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div
      style={{
        display: "flex",
        width: "100vw",
        height: "100vh",
        background: "#000",
        fontFamily: "'Courier New', monospace",
        color: "#E0E0FF",
        position: "relative"
      }}
    >
      {/* Back button */}
      <button 
        onClick={() => setActiveScreen("selection")}
        style={{
          position: "absolute", top: "20px", left: "20px", zIndex: 10,
          background: "rgba(0,0,0,0.8)", border: "1px solid #FF00AA", color: "#FF00AA",
          padding: "8px 16px", cursor: "pointer", fontFamily: "'Courier New', monospace",
          transition: "all 0.2s"
        }}
        onMouseOver={(e) => { e.target.style.background = "rgba(255, 0, 170, 0.2)" }}
        onMouseOut={(e) => { e.target.style.background = "rgba(0,0,0,0.8)" }}
      >
        ← Change Model
      </button>

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
