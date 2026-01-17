import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


async def send_event(event_type: str, payload: dict) -> None:
    if not settings.gumloop_webhook_url:
        return
    body = {"type": event_type, **payload}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(settings.gumloop_webhook_url, json=body)
    except Exception as exc:
        logger.warning("Gumloop event failed: %s", exc)


async def send_session_complete(session_id: str, summary: dict) -> None:
    await send_event("SESSION_COMPLETE", {"session_id": session_id, "summary": summary})
