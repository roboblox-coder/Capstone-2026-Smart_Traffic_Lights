"""WebSocket Server for SUMO Simulation.

Reusable WebSocket server that runs in a background thread, broadcasts
simulation data to all connected clients, and collects incoming commands
for the simulation loop.

Usage:
    server = SimulationWebSocketServer(host="localhost", port=8765)
    server.start()                       # non-blocking
    server.broadcast(data)               # send dict to all clients
    cmds = server.get_pending_commands() # drain incoming queue
    server.stop()                        # graceful shutdown
"""

from __future__ import annotations

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

        self._clients: set = set()
        self._command_queue: deque = deque()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server = None
        self._ready = threading.Event()

    # ── public API ──────────────────────────────────────────────────

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        # Wait for the event loop AND the listener to be live before returning.
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("WebSocket server failed to start within 5s")
        print(f"[ws] WebSocket server listening on ws://{self.host}:{self.port}")

    def stop(self) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=3)
        print("[ws] WebSocket server stopped.")

    def broadcast(self, data: dict) -> None:
        if self._loop is None or not self._loop.is_running():
            return
        try:
            message = json.dumps(data)
        except (TypeError, ValueError) as e:
            print(f"   [warn] broadcast: non-JSON-serialisable payload ({e})")
            return
        asyncio.run_coroutine_threadsafe(self._broadcast_async(message), self._loop)

    def get_pending_commands(self) -> list[dict]:
        commands: list[dict] = []
        while self._command_queue:
            commands.append(self._command_queue.popleft())
        return commands

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # ── internals ───────────────────────────────────────────────────

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        async with serve(self._handle_client, self.host, self.port) as server:
            self._server = server
            # Only signal "ready" once the listener is actually accepting.
            self._ready.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                pass

    async def _handle_client(self, websocket) -> None:
        self._clients.add(websocket)
        remote = websocket.remote_address
        print(f"   [ws] Client connected: {remote}  (total={len(self._clients)})")
        try:
            async for raw_message in websocket:
                try:
                    cmd = json.loads(raw_message)
                    self._command_queue.append(cmd)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON",
                    }))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)
            print(f"   [ws] Client disconnected: {remote}  (total={len(self._clients)})")

    async def _broadcast_async(self, message: str) -> None:
        # Snapshot to avoid "set changed size during iteration" if a client
        # disconnects mid-broadcast.
        clients = tuple(self._clients)
        if not clients:
            return
        await asyncio.gather(
            *(c.send(message) for c in clients),
            return_exceptions=True,
        )
