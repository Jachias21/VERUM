"""
Rate limiter backed by MongoDB.

Uses a `rate_limits` collection with documents {user_hash, window_start, count}.
A TTL index on `window_start` (expireAfterSeconds=120) keeps the collection clean
without any manual housekeeping.

Environment variables:
  RATE_LIMIT_MAX_REQUESTS   — max requests per window (default: 10)
  RATE_LIMIT_WINDOW_SECONDS — length of the sliding window in seconds (default: 60)
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

from pymongo import ReturnDocument

from shared.db import get_mongo_db

logger = logging.getLogger(__name__)

_COLLECTION = "rate_limits"
_index_ensured = False


async def _ensure_ttl_index() -> None:
    """Create the TTL index on first use (idempotent — MongoDB ignores duplicate index calls)."""
    global _index_ensured
    if _index_ensured:
        return
    try:
        db = get_mongo_db()
        await db[_COLLECTION].create_index("window_start", expireAfterSeconds=120, background=True)
        _index_ensured = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("rate_limiter: could not ensure TTL index — %s", exc)


async def check_rate_limit(user_hash: str) -> bool:
    """Return True if the request is within limits, False if the user is rate-limited.

    Fails open: if MongoDB is unavailable, returns True so legitimate users are
    not blocked by infrastructure problems.
    """
    max_requests = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "10"))
    window_seconds = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=window_seconds)

    try:
        await _ensure_ttl_index()

        db = get_mongo_db()
        collection = db[_COLLECTION]

        # Atomically increment count if a document for this user exists within the
        # current window. Returns the updated document (after update).
        doc = await collection.find_one_and_update(
            {"user_hash": user_hash, "window_start": {"$gte": cutoff}},
            {"$inc": {"count": 1}},
            return_document=ReturnDocument.AFTER,
            upsert=False,
        )

        if doc is None:
            # No in-window document: start a fresh window for this user.
            await collection.update_one(
                {"user_hash": user_hash},
                {"$set": {"window_start": now, "count": 1}},
                upsert=True,
            )
            return True

        allowed = doc["count"] <= max_requests
        if not allowed:
            logger.warning(
                "rate_limiter: user_hash=%.12s… exceeded %d requests in %ds (count=%d)",
                user_hash, max_requests, window_seconds, doc["count"],
            )
        return allowed

    except Exception as exc:  # noqa: BLE001
        logger.error("rate_limiter: MongoDB error — failing open. %s: %s", type(exc).__name__, exc)
        return True  # fail open
