import { useEffect, useState, useRef } from "react";
import { io } from "socket.io-client";

const SOCKET_URL = "http://localhost:8000";

/**
 * useSocket — Custom hook for Socket.IO connection to the NeonDrift
 * inference server.
 *
 * Returns:
 *   telemetry : object | null — latest telemetry payload from server
 *   connected : boolean       — WebSocket connection status
 *
 * Guards against React StrictMode double-render by storing the socket
 * in a ref and checking a mounted flag.
 */
export function useSocket() {
  const [telemetry, setTelemetry] = useState(null);
  const [connected, setConnected] = useState(false);
  const socketRef = useRef(null);
  const mountedRef = useRef(false);

  useEffect(() => {
    // Guard against StrictMode double-invoke
    if (mountedRef.current) return;
    mountedRef.current = true;

    const socket = io(SOCKET_URL, { transports: ["websocket"] });
    socketRef.current = socket;

    socket.on("connect", () => setConnected(true));
    socket.on("disconnect", () => setConnected(false));
    socket.on("telemetry", (data) => setTelemetry(data));
    socket.on("reset", () => setTelemetry(null));

    return () => {
      socket.disconnect();
      mountedRef.current = false;
    };
  }, []);

  return { telemetry, connected };
}
