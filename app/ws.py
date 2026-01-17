import asyncio
from typing import Dict, Set

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.setdefault(session_id, set()).add(websocket)

    async def disconnect(self, session_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            sessions = self._connections.get(session_id)
            if not sessions:
                return
            sessions.discard(websocket)
            if not sessions:
                self._connections.pop(session_id, None)

    async def broadcast_session(self, session_id: str, payload: dict) -> None:
        async with self._lock:
            sockets = list(self._connections.get(session_id, set()))
        for websocket in sockets:
            try:
                await websocket.send_json(payload)
            except Exception:
                await self.disconnect(session_id, websocket)
