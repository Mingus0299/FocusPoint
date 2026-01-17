from motor.motor_asyncio import AsyncIOMotorClient

from app.config import settings

_client: AsyncIOMotorClient | None = None
_db = None


def get_db():
    return _db


async def connect() -> None:
    global _client, _db
    _client = AsyncIOMotorClient(settings.mongo_uri)
    _db = _client[settings.mongo_db]


async def disconnect() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
