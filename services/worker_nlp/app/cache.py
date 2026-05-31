"""
Verdict cache — stores NLPResult verdicts in MongoDB with a TTL.
Only FAKE and REAL verdicts are cached (UNVERIFIED is too ambiguous to reuse).

The cache key is the SHA-256 hash of the normalised input text, so the
original message is never stored.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from shared.db import get_mongo_db
from shared.schemas import NLPResult

logger = logging.getLogger("verum.cache")

_COLLECTION = "verdict_cache"
_INDEX_CREATED = False


async def _ensure_ttl_index() -> None:
    """Create the TTL index on first use (idempotent)."""
    global _INDEX_CREATED
    if _INDEX_CREATED:
        return
    try:
        db = get_mongo_db()
        await db[_COLLECTION].create_index("expires_at", expireAfterSeconds=0)
        _INDEX_CREATED = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("[cache] Could not create TTL index: %s", exc)


async def get_cached_verdict(text_hash: str) -> NLPResult | None:
    """Return a cached NLPResult for *text_hash*, or None on miss / error."""
    try:
        await _ensure_ttl_index()
        db = get_mongo_db()
        now = datetime.now(timezone.utc)
        doc = await db[_COLLECTION].find_one(
            {"text_hash": text_hash, "expires_at": {"$gt": now}}
        )
        if doc is None:
            return None
        result = NLPResult.model_validate(doc["result"])
        logger.info("[cache] HIT text_hash=%.16s…", text_hash)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("[cache] get_cached_verdict failed (cache bypassed): %s", exc)
        return None


async def set_cached_verdict(
    text_hash: str, result: NLPResult, ttl_hours: int = 24
) -> None:
    """Persist *result* in the cache keyed by *text_hash* (FAKE/REAL only)."""
    if result.verdict not in ("FAKE", "REAL"):
        return
    try:
        await _ensure_ttl_index()
        db = get_mongo_db()
        expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
        await db[_COLLECTION].update_one(
            {"text_hash": text_hash},
            {"$set": {"result": result.model_dump(mode="json"), "expires_at": expires_at}},
            upsert=True,
        )
        logger.info("[cache] SET text_hash=%.16s… expires_at=%s", text_hash, expires_at.isoformat())
    except Exception as exc:  # noqa: BLE001
        logger.warning("[cache] set_cached_verdict failed (cache bypassed): %s", exc)
