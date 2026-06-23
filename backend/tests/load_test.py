"""
NeonDrift — WebSocket Load Testing Script.

Simulates N concurrent clients connecting to the inference server and measures
the effective Frames Per Second (FPS) received by each client. 

Usage:
    python inference/load_test.py --clients 50 --duration 10
"""

import argparse
import asyncio
import time
import socketio

async def simulate_client(client_id: int, url: str, duration: int, stats: dict):
    """Simulate a single client connection and count telemetry events."""
    sio = socketio.AsyncClient(logger=False, engineio_logger=False)
    
    # Track metrics for this client
    client_stats = {
        "telemetry_count": 0,
        "connected": False,
        "error": None
    }
    stats[client_id] = client_stats

    @sio.event
    async def connect():
        client_stats["connected"] = True

    @sio.event
    async def disconnect():
        client_stats["connected"] = False

    @sio.event
    async def telemetry(data):
        client_stats["telemetry_count"] += 1

    try:
        await asyncio.sleep(client_id * 0.05)
        await sio.connect(url, transports=["websocket"], wait_timeout=30)
            
        await asyncio.sleep(duration)
    except Exception as e:
        client_stats["error"] = str(e)
    finally:
        if sio.connected:
            await sio.disconnect()

async def main():
    parser = argparse.ArgumentParser(description="Load test the NeonDrift inference server.")
    parser.add_argument("--clients", type=int, default=50, help="Number of concurrent clients (default: 50)")
    parser.add_argument("--duration", type=int, default=10, help="Duration of the test in seconds (default: 10)")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Server URL (default: http://localhost:8000)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  NeonDrift Load Tester")
    print(f"  Target URL : {args.url}")
    print(f"  Clients    : {args.clients}")
    print(f"  Duration   : {args.duration} seconds")
    print("=" * 60)
    print("Spawning clients...")

    # Shared dictionary to aggregate results
    stats = {}
    
    start_time = time.time()
    
    # Spawn all clients concurrently
    tasks = [
        simulate_client(i, args.url, args.duration, stats) 
        for i in range(args.clients)
    ]
    await asyncio.gather(*tasks)
    
    elapsed = time.time() - start_time

    # Analyze results
    total_telemetry = 0
    connected_clients = 0
    errors = 0

    for client_id, s in stats.items():
        total_telemetry += s["telemetry_count"]
        if s["error"] is not None:
            errors += 1
        elif s["telemetry_count"] > 0:
            connected_clients += 1

    # Calculate metrics
    avg_fps = 0.0
    if connected_clients > 0:
        avg_fps = (total_telemetry / connected_clients) / args.duration

    print("\n" + "=" * 60)
    print("  Load Test Results")
    print("=" * 60)
    print(f"  Total Clients Spawned : {args.clients}")
    print(f"  Successful Clients    : {connected_clients}")
    print(f"  Failed Clients        : {errors}")
    for client_id, s in stats.items():
        if s["error"] is not None:
            print(f"  [ERROR Sample] Client {client_id}: {s['error']}")
            break
    print(f"  Total Telemetry Msgs  : {total_telemetry}")
    print("-" * 60)
    print(f"  Target FPS per client : ~30.00")
    print(f"  Actual FPS per client : {avg_fps:.2f}  <-- THE BENCHMARK")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
