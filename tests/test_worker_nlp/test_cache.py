"""
Tests for services/worker_nlp/app/cache.py

MongoDB is mocked completely — no running database required.
The entire module is skipped when motor is not installed (outside Docker).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip every test in this module when motor (async MongoDB driver) is absent
pytest.importorskip("motor")

from shared.schemas import NLPResult  # noqa: E402 — after importorskip guard


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_nlp_result(verdict: str = "FAKE") -> NLPResult:
    return NLPResult(
        query_id=uuid.uuid4(),
        extracted_entities=["Madrid", "España"],
        fact_check_matches=2,
        source_url="https://example.com/article",
        verdict=verdict,  # type: ignore[arg-type]
        retrieved_context="Texto de contexto de prueba.",
        summary="VEREDICTO: FALSO — El artículo confirma el bulo.",
    )


def _make_mock_collection() -> MagicMock:
    """Return a MagicMock whose async methods are AsyncMocks."""
    col = MagicMock()
    col.find_one = AsyncMock()
    col.update_one = AsyncMock()
    col.create_index = AsyncMock()
    return col


def _make_mock_db(collection: MagicMock) -> MagicMock:
    db = MagicMock()
    db.__getitem__ = MagicMock(return_value=collection)
    return db


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_index_flag():
    """Reset the _INDEX_CREATED module flag before and after every test."""
    import services.worker_nlp.app.cache as cache_mod
    cache_mod._INDEX_CREATED = False
    yield
    cache_mod._INDEX_CREATED = False


# ── Test 1: cache HIT — existing, non-expired document → NLPResult ───────────

async def test_get_cached_verdict_hit():
    from services.worker_nlp.app.cache import get_cached_verdict

    expected = _make_nlp_result("FAKE")
    doc = {
        "text_hash": "deadbeef1234",
        "result": expected.model_dump(mode="json"),
        "expires_at": datetime(2099, 1, 1, tzinfo=timezone.utc),
    }

    col = _make_mock_collection()
    col.find_one = AsyncMock(return_value=doc)
    mock_db = _make_mock_db(col)

    with patch("services.worker_nlp.app.cache.get_mongo_db", return_value=mock_db):
        result = await get_cached_verdict("deadbeef1234")

    assert result is not None
    assert result.verdict == "FAKE"
    assert "Madrid" in result.extracted_entities
    col.find_one.assert_awaited_once()


# ── Test 2: cache MISS — key not present in collection → None ────────────────

async def test_get_cached_verdict_miss():
    from services.worker_nlp.app.cache import get_cached_verdict

    col = _make_mock_collection()
    col.find_one = AsyncMock(return_value=None)  # document not found
    mock_db = _make_mock_db(col)

    with patch("services.worker_nlp.app.cache.get_mongo_db", return_value=mock_db):
        result = await get_cached_verdict("nonexistent_key_xyz")

    assert result is None


# ── Test 3: set_cached_verdict with FAKE verdict → update_one is called ───────

async def test_set_cached_verdict_fake_stores_in_mongodb():
    from services.worker_nlp.app.cache import set_cached_verdict

    nlp_result = _make_nlp_result("FAKE")
    col = _make_mock_collection()
    mock_db = _make_mock_db(col)

    with patch("services.worker_nlp.app.cache.get_mongo_db", return_value=mock_db):
        await set_cached_verdict("abc123fake", nlp_result)

    col.update_one.assert_awaited_once()
    # First positional argument to update_one must be the filter by text_hash
    filter_arg = col.update_one.call_args[0][0]
    assert filter_arg == {"text_hash": "abc123fake"}
    # The call must request an upsert
    kwargs = col.update_one.call_args[1]
    assert kwargs.get("upsert") is True


# ── Test 4: set_cached_verdict with UNVERIFIED → update_one never called ──────

async def test_set_cached_verdict_unverified_skips_storage():
    """UNVERIFIED verdicts must be silently ignored — too ambiguous to cache."""
    from services.worker_nlp.app.cache import set_cached_verdict

    nlp_result = _make_nlp_result("UNVERIFIED")
    col = _make_mock_collection()
    mock_db = _make_mock_db(col)

    with patch("services.worker_nlp.app.cache.get_mongo_db", return_value=mock_db):
        await set_cached_verdict("xyz789unverified", nlp_result)

    col.update_one.assert_not_awaited()


# ── Test 5: get_cached_verdict when MongoDB raises → fail-open (None) ─────────

async def test_get_cached_verdict_mongodb_failure_returns_none():
    """Any MongoDB error must be swallowed and None returned (fail-open)."""
    from services.worker_nlp.app.cache import get_cached_verdict

    col = _make_mock_collection()
    col.find_one = AsyncMock(side_effect=Exception("Connection refused by MongoDB"))
    mock_db = _make_mock_db(col)

    with patch("services.worker_nlp.app.cache.get_mongo_db", return_value=mock_db):
        result = await get_cached_verdict("some_hash_value")

    assert result is None
