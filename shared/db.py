"""
Shared database client factories.
Each service calls these helpers to obtain ready-to-use clients.
"""
from __future__ import annotations

import os

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase


def get_mongo_db() -> AsyncIOMotorDatabase:
    uri = (
        f"mongodb://{os.environ['MONGO_USER']}:{os.environ['MONGO_PASS']}"
        f"@{os.environ['MONGO_HOST']}:{os.environ['MONGO_PORT']}"
    )
    client = AsyncIOMotorClient(uri)
    return client[os.environ["MONGO_DB"]]


def get_qdrant_client():
    from qdrant_client import QdrantClient  # lazy import — not available in worker_vision
    return QdrantClient(
        host=os.environ["QDRANT_HOST"],
        port=int(os.environ["QDRANT_PORT"]),
    )
