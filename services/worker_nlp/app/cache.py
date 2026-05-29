"""
Caché de veredictos - almacena NLPResult en MongoDB con TTL.
Solo se cachean veredictos FAKE y REAL (UNVERIFIED es demasiado ambiguo para reutilizar).

La clave es el hash SHA-256 del texto de entrada normalizado;
el mensaje original nunca se almacena.
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
    """Crea el índice TTL en el primer uso (idempotente)."""
    global _INDEX_CREATED
    if _INDEX_CREATED:
        return
    try:
        db = get_mongo_db()
        await db[_COLLECTION].create_index("expires_at", expireAfterSeconds=0)
        _INDEX_CREATED = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("[cache] No se pudo crear el índice TTL: %s", exc)


async def get_cached_verdict(text_hash: str) -> NLPResult | None:
    """Devuelve un NLPResult cacheado para *text_hash*, o None si no existe o hay error."""
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
        logger.info("[cache] HIT text_hash=%.16s...", text_hash)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("[cache] get_cached_verdict falló (caché ignorada): %s", exc)
        return None


async def set_cached_verdict(
    text_hash: str, result: NLPResult, ttl_hours: int = 24
) -> None:
    """Persiste *result* en la caché indexado por *text_hash* (solo FAKE/REAL)."""
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
        logger.info("[cache] SET text_hash=%.16s... expires_at=%s", text_hash, expires_at.isoformat())
    except Exception as exc:  # noqa: BLE001
        logger.warning("[cache] set_cached_verdict falló (caché ignorada): %s", exc)
