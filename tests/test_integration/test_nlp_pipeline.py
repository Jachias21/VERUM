"""
Integration tests for the NLP worker pipeline end-to-end.

All external dependencies (Qdrant, Ollama, Telegram, MongoDB) are replaced
with async/sync mocks so the suite runs without Docker.

Run with: pytest tests/test_integration/ -v
"""
from __future__ import annotations

import datetime
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.schemas import NLPResult, TextTask

# ── Shared helpers ─────────────────────────────────────────────────────────────

_VIRAL_TEXT = (
    "El Gobierno de España ha confirmado que el nuevo virus detectado en Madrid "
    "no representa ningún peligro real para la población según fuentes oficiales."
)


def _make_task(text: str = _VIRAL_TEXT, chat_id: int = 42) -> TextTask:
    return TextTask(
        query_id=uuid.uuid4(),
        user_hash="aabbccdd" + "0" * 56,  # 64-char SHA-256-shaped string
        chat_id=chat_id,
        text=text,
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )


class _AsyncNoopContext:
    """Async context manager that does nothing — simulates message.process()."""

    async def __aenter__(self):
        return None

    async def __aexit__(self, *_):
        return None


class FakeMessage:
    """Minimal stub for aio_pika.abc.AbstractIncomingMessage."""

    def __init__(self, task: TextTask) -> None:
        self.body = task.model_dump_json().encode()

    def process(self) -> _AsyncNoopContext:
        return _AsyncNoopContext()


def _make_mongo_mock() -> MagicMock:
    """Return a motor-compatible database mock with an async insert_one."""
    collection = MagicMock()
    collection.insert_one = AsyncMock()
    db = MagicMock()
    db.__getitem__ = MagicMock(return_value=collection)
    return db


def _sent_text(mock_send: AsyncMock) -> str:
    """Extract the 'text' kwarg from the first Bot.send_message call."""
    return mock_send.call_args.kwargs["text"]


# ── Test 1: high-confidence L1 Qdrant hit → FAKE verdict ─────────────────────

@pytest.mark.asyncio
async def test_full_pipeline_fake_verdict():
    from services.worker_nlp.app.worker import process

    task = _make_task()
    message = FakeMessage(task)

    qdrant_hit = [
        {
            "score": 0.82,
            "verdict": "FAKE",
            "text": "El bulo sobre el virus fue desmentido. La afirmación es completamente falsa.",
            "url": "https://maldita.es/malditaciencia/fake-virus",
        }
    ]

    with (
        patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "fake_token"}),
        patch("services.worker_nlp.app.worker.is_gibberish", return_value=False),
        patch("services.worker_nlp.app.worker.get_cached_verdict", new=AsyncMock(return_value=None)),
        patch("services.worker_nlp.app.worker.extract_entities", return_value=["Gobierno", "Madrid", "virus"]),
        patch("services.worker_nlp.app.rag._search_qdrant", new=AsyncMock(return_value=qdrant_hit)),
        patch(
            "services.worker_nlp.app.worker.synthesize_verdict",
            new=AsyncMock(return_value="VEREDICTO: FALSO — el artículo lo confirma."),
        ),
        patch("services.worker_nlp.app.worker.set_cached_verdict", new=AsyncMock()),
        patch("services.worker_nlp.app.worker.get_mongo_db", return_value=_make_mongo_mock()),
        patch("services.worker_nlp.app.worker.Bot") as MockBot,
    ):
        mock_send = AsyncMock()
        MockBot.return_value.send_message = mock_send
        await process(message)

    mock_send.assert_called_once()
    text = _sent_text(mock_send)
    assert "🔴" in text
    assert "FAKE" in text


# ── Test 2: all sources empty → UNVERIFIED + maldita.es fallback ─────────────

@pytest.mark.asyncio
async def test_full_pipeline_unverified_no_hits():
    """Zero-day hoax: no Qdrant hit, no Google FC, no GNews → UNVERIFIED reply."""
    from services.worker_nlp.app.worker import process

    task = _make_task()
    message = FakeMessage(task)

    with (
        patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "fake_token"}),
        patch("services.worker_nlp.app.worker.is_gibberish", return_value=False),
        patch("services.worker_nlp.app.worker.get_cached_verdict", new=AsyncMock(return_value=None)),
        patch("services.worker_nlp.app.worker.extract_entities", return_value=["virus", "Madrid"]),
        patch("services.worker_nlp.app.rag._search_qdrant", new=AsyncMock(return_value=[])),
        patch("services.worker_nlp.app.rag._search_google_fact_check", new=AsyncMock(return_value=[])),
        patch("services.worker_nlp.app.rag._search_gnews", new=AsyncMock(return_value=[])),
        patch(
            "services.worker_nlp.app.worker.synthesize_verdict",
            new=AsyncMock(return_value="No tengo información sobre esta afirmación."),
        ),
        patch("services.worker_nlp.app.worker.set_cached_verdict", new=AsyncMock()),
        patch("services.worker_nlp.app.worker.get_mongo_db", return_value=_make_mongo_mock()),
        patch("services.worker_nlp.app.worker.Bot") as MockBot,
    ):
        mock_send = AsyncMock()
        MockBot.return_value.send_message = mock_send
        await process(message)

    mock_send.assert_called_once()
    text = _sent_text(mock_send)
    assert "🟡" in text
    assert any(site in text for site in ("maldita.es", "newtral.es", "snopes.com"))


# ── Test 3: gibberish input → pipeline aborted, no Qdrant call ───────────────

@pytest.mark.asyncio
async def test_full_pipeline_gibberish_rejected():
    """Incoherent input must short-circuit at NER without touching Qdrant."""
    from services.worker_nlp.app.worker import process

    task = _make_task(text="asdf qwer zxcv 1234 !! foo bar baz")
    message = FakeMessage(task)

    mock_qdrant = AsyncMock()

    with (
        patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "fake_token"}),
        patch("services.worker_nlp.app.worker.is_gibberish", return_value=True),
        patch("services.worker_nlp.app.rag._search_qdrant", new=mock_qdrant),
        patch("services.worker_nlp.app.worker.Bot") as MockBot,
    ):
        mock_send = AsyncMock()
        MockBot.return_value.send_message = mock_send
        await process(message)

    mock_qdrant.assert_not_called()
    mock_send.assert_called_once()
    text = _sent_text(mock_send)
    assert "⚠️" in text or "entender" in text


# ── Test 4: cached verdict → pipeline short-circuited, no Qdrant call ─────────

@pytest.mark.asyncio
async def test_full_pipeline_cache_hit():
    """A cache hit must skip the search pipeline and add the ⚡ indicator."""
    from services.worker_nlp.app.worker import process

    task = _make_task()
    message = FakeMessage(task)

    cached_result = NLPResult(
        query_id=task.query_id,
        extracted_entities=["Madrid", "virus"],
        fact_check_matches=1,
        source_url="https://maldita.es/cached",
        verdict="FAKE",
        retrieved_context="",
        summary="VEREDICTO: FALSO — ya registrado en caché.",
    )
    mock_qdrant = AsyncMock()

    with (
        patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "fake_token"}),
        patch("services.worker_nlp.app.worker.is_gibberish", return_value=False),
        patch(
            "services.worker_nlp.app.worker.get_cached_verdict",
            new=AsyncMock(return_value=cached_result),
        ),
        patch("services.worker_nlp.app.rag._search_qdrant", new=mock_qdrant),
        patch("services.worker_nlp.app.worker.get_mongo_db", return_value=_make_mongo_mock()),
        patch("services.worker_nlp.app.worker.Bot") as MockBot,
    ):
        mock_send = AsyncMock()
        MockBot.return_value.send_message = mock_send
        await process(message)

    mock_qdrant.assert_not_called()
    mock_send.assert_called_once()
    text = _sent_text(mock_send)
    assert "⚡" in text
