"""
WebSocket Server for SUMO Simulation
=====================================
Reusable WebSocket server that runs in a background thread.
Broadcasts simulation data to all connected clients and
collects incoming commands for the simulation loop.

Usage:
    server = SimulationWebSocketServer(host="localhost", port=8765)
    server.start()          # non-blocking, spins up a background thread

    server.broadcast(data)  # send dict to all clients  (thread-safe)
    cmds = server.get_pending_commands()  # drain incoming command queue

    server.stop()           # graceful shutdown
"""

import asyncio
import json
import threading
from collections import deque

try:
    import websockets
    from websockets.asyncio.server import serve
except ImportError:
    raise ImportError(
        "The 'websockets' package is required.  Install it with:\n"
        "  pip install websockets"
    )


class SimulationWebSocketServer:
    """Threaded WebSocket server for SUMO ↔ frontend communication."""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port

        # Connected client websockets
        self._clients: set = set()

        # Thread-safe queue for commands received from clients
        self._command_queue: deque = deque()

        # Internal event-loop / thread handles
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server = None

    # ── public API (called from the simulation thread) ──────────────

    def start(self) -> None:
        """Start the server in a background daemon thread."""
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        # Wait briefly for the loop to become available
        for _ in range(50):
            if self._loop is not None and self._loop.is_running():
                break
            import time
            time.sleep(0.05)
        print(f"🌐 WebSocket server listening on ws://{self.host}:{self.port}")

    def stop(self) -> None:
        """Gracefully shut down the server."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=3)
        print("🌐 WebSocket server stopped.")

    def broadcast(self, data: dict) -> None:
        """Send *data* (as JSON) to every connected client."""
        if not self._clients or self._loop is None:
            return
        message = json.dumps(data)
        # Schedule sends on the event-loop thread
        asyncio.run_coroutine_threadsafe(self._broadcast_async(message), self._loop)

    def get_pending_commands(self) -> list[dict]:
        """Drain and return all commands received since the last call."""
        commands: list[dict] = []
        while self._command_queue:
            commands.append(self._command_queue.popleft())
        return commands

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # ── internals ───────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Entry-point for the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        async with serve(self._handle_client, self.host, self.port) as server:
            self._server = server
            await asyncio.Future()  # run forever

    async def _handle_client(self, websocket) -> None:
        self._clients.add(websocket)
        remote = websocket.remote_address
        print(f"   🔗 Client connected: {remote}")
        try:
            async for raw_message in websocket:
                try:
                    cmd = json.loads(raw_message)
                    self._command_queue.append(cmd)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON"
                    }))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)
            print(f"   🔌 Client disconnected: {remote}")

    async def _broadcast_async(self, message: str) -> None:
        if self._clients:
            await asyncio.gather(
                *(client.send(message) for client in self._clients),
                return_exceptions=True,
            )
