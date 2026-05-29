"""
Rate limiter respaldado por MongoDB.

Usa una colección `rate_limits` con documentos {user_hash, window_start, count}.
Un índice TTL sobre `window_start` (expireAfterSeconds=120) mantiene limpia la
colección sin mantenimiento manual.

Variables de entorno:
  RATE_LIMIT_MAX_REQUESTS   - máximo de peticiones por ventana (por defecto: 10)
  RATE_LIMIT_WINDOW_SECONDS - duración de la ventana deslizante en segundos (por defecto: 60)
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
    """Crea el índice TTL en el primer uso (idempotente - MongoDB ignora llamadas de índice duplicadas)."""
    global _index_ensured
    if _index_ensured:
        return
    try:
        db = get_mongo_db()
        await db[_COLLECTION].create_index("window_start", expireAfterSeconds=120, background=True)
        _index_ensured = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("rate_limiter: no se pudo asegurar el índice TTL - %s", exc)


async def check_rate_limit(user_hash: str) -> bool:
    """Devuelve True si la petición está dentro del límite, False si el usuario ha sido limitado.
    Falla abierto: si MongoDB no está disponible, devuelve True para no bloquear a usuarios legítimos.
    """
    max_requests = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "10"))
    window_seconds = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=window_seconds)

    try:
        await _ensure_ttl_index()

        db = get_mongo_db()
        collection = db[_COLLECTION]

        doc = await collection.find_one_and_update(
            {"user_hash": user_hash, "window_start": {"$gte": cutoff}},
            {"$inc": {"count": 1}},
            return_document=ReturnDocument.AFTER,
            upsert=False,
        )

        if doc is None:
            await collection.update_one(
                {"user_hash": user_hash},
                {"$set": {"window_start": now, "count": 1}},
                upsert=True,
            )
            return True

        allowed = doc["count"] <= max_requests
        if not allowed:
            logger.warning(
                "rate_limiter: user_hash=%.12s... superó %d peticiones en %ds (count=%d)",
                user_hash, max_requests, window_seconds, doc["count"],
            )
        return allowed

    except Exception as exc:  # noqa: BLE001
        logger.error("rate_limiter: error MongoDB - falla abierto. %s: %s", type(exc).__name__, exc)
        return True  # falla abierto
