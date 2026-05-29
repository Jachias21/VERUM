"""
Tests de integración para el pipeline del worker NLP de extremo a extremo.

Todas las dependencias externas (Qdrant, Ollama, Telegram, MongoDB) se sustituyen
con mocks async/sync para que el suite funcione sin Docker.

Ejecutar con: pytest tests/test_integration/ -v
"""
from __future__ import annotations

import asyncio
import datetime
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("aio_pika", reason="aio_pika not installed (Docker-only dependency)")

from shared.schemas import NLPResult, TextTask

# ── Shared helpers ─────────────────────────────────────────────────────────────

_VIRAL_TEXT = (
    "El Gobierno de España ha confirmado que el nuevo virus detectado en Madrid "
    "no representa ningún peligro real para la población según fuentes oficiales."
)


def _make_task(text: str = _VIRAL_TEXT, chat_id: int = 42) -> TextTask:
    return TextTask(
        query_id=uuid.uuid4(),
    user_hash="aabbccdd" + "0" * 56,  # cadena de 64 chars con forma SHA-256
        chat_id=chat_id,
        text=text,
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )


class _AsyncNoopContext:
    """Gestor de contexto async que no hace nada — simula message.process()."""

    async def __aenter__(self):
        return None

    async def __aexit__(self, *_):
        return None


class FakeMessage:
    """Stub mínimo para aio_pika.abc.AbstractIncomingMessage."""

    def __init__(self, task: TextTask) -> None:
        self.body = task.model_dump_json().encode()

    def process(self) -> _AsyncNoopContext:
        return _AsyncNoopContext()


def _make_mongo_mock() -> MagicMock:
    """Devuelve un mock de base de datos compatible con motor con insert_one async."""
    collection = MagicMock()
    collection.insert_one = AsyncMock()
    db = MagicMock()
    db.__getitem__ = MagicMock(return_value=collection)
    return db


def _make_mongo_mock_failing() -> MagicMock:
    """Devuelve un mock MongoDB donde insert_one lanza para simular un fallo de conectividad."""
    collection = MagicMock()
    collection.insert_one = AsyncMock(side_effect=Exception("MongoDB unreachable"))
    db = MagicMock()
    db.__getitem__ = MagicMock(return_value=collection)
    return db


def _sent_text(mock_send: AsyncMock) -> str:
    """Extrae el kwarg 'text' de la primera llamada a Bot.send_message."""
    return mock_send.call_args.kwargs["text"]


def _make_synth_mock(summary: str) -> AsyncMock:
    """Devuelve un AsyncMock para synthesize_verdict que establece result.summary en el objeto
    y devuelve el NLPResult modificado — replicando la API de la función real."""
    async def _side_effect(result, text):  # noqa: ANN001
        result.summary = summary
        return result
    return AsyncMock(side_effect=_side_effect)


# ── Test 1: hit L1 Qdrant de alta confianza → veredicto FAKE ─────────────────

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
            new=_make_synth_mock("VEREDICTO: FALSO — el artículo lo confirma."),
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


# ── Test 2: todas las fuentes vacías → UNVERIFIED + fallback maldita.es ────────

@pytest.mark.asyncio
async def test_full_pipeline_unverified_no_hits():
    """Bulo de cer-o día: sin hit en Qdrant, Google FC ni GNews → respuesta UNVERIFIED."""
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
            new=_make_synth_mock("No tengo información sobre esta afirmación."),
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


# ── Test 3: entrada sin sentido → pipeline abortado, sin llamada a Qdrant ─────

@pytest.mark.asyncio
async def test_full_pipeline_gibberish_rejected():
    """Una entrada incoherente debe cortocircuitar en NER sin tocar Qdrant."""
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


# ── Test 4: veredicto en caché → pipeline cortocircuitado, sin llamada a Qdrant ──

@pytest.mark.asyncio
async def test_full_pipeline_cache_hit():
    """Un hit en caché debe omitir el pipeline de búsqueda y añadir el indicador ⚡."""
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


# ── Test 5: timeout de Ollama → worker lo captura, envía ⚠️, sin crash ───────

@pytest.mark.asyncio
async def test_pipeline_ollama_timeout_sends_error_reply():
    """asyncio.TimeoutError lanzado por synthesize_verdict debe capturarse;
    el usuario recibe ⚠️ y el worker no propaga la excepción."""
    from services.worker_nlp.app.worker import process

    task = _make_task()
    message = FakeMessage(task)

    mock_extract = MagicMock(return_value=["Gobierno", "Madrid"])
    rag_result = NLPResult(
        query_id=task.query_id,
        extracted_entities=["Gobierno", "Madrid"],
        fact_check_matches=0,
        source_url=None,
        verdict="UNVERIFIED",
        retrieved_context="",
        summary="",
    )

    with (
        patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "fake_token"}),
        patch("services.worker_nlp.app.worker.is_gibberish", return_value=False),
        patch("services.worker_nlp.app.worker.get_cached_verdict", new=AsyncMock(return_value=None)),
        patch("services.worker_nlp.app.worker.extract_entities", new=mock_extract),
        patch(
            "services.worker_nlp.app.worker.hybrid_search",
            new=AsyncMock(return_value=rag_result),
        ),
        patch(
            "services.worker_nlp.app.worker.synthesize_verdict",
            new=AsyncMock(side_effect=asyncio.TimeoutError()),
        ),
        patch("services.worker_nlp.app.worker.Bot") as MockBot,
    ):
        mock_send = AsyncMock()
        MockBot.return_value.send_message = mock_send
        await process(message)  # must not raise

    mock_extract.assert_called_once()
    mock_send.assert_called_once()
    text = _sent_text(mock_send)
    assert "⚠️" in text


# ── Test 6: salida LLM malformada → veredicto RAG preservado ───────────────────

@pytest.mark.asyncio
async def test_pipeline_malformed_llm_keeps_rag_verdict():
    """La salida del LLM sin etiqueta VEREDICTO: no debe sobreescribir el veredicto RAG."""
    from services.worker_nlp.app.worker import process

    task = _make_task()
    message = FakeMessage(task)

    rag_result = NLPResult(
        query_id=task.query_id,
        extracted_entities=["Gobierno", "Madrid"],
        fact_check_matches=1,
        source_url="https://maldita.es/test",
        verdict="FAKE",
        retrieved_context="Contexto de la base de conocimiento.",
        summary="",
    )

    with (
        patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "fake_token"}),
        patch("services.worker_nlp.app.worker.is_gibberish", return_value=False),
        patch("services.worker_nlp.app.worker.get_cached_verdict", new=AsyncMock(return_value=None)),
        patch("services.worker_nlp.app.worker.extract_entities", return_value=["Gobierno", "Madrid"]),
        patch(
            "services.worker_nlp.app.worker.hybrid_search",
            new=AsyncMock(return_value=rag_result),
        ),
        patch(
            "services.worker_nlp.app.worker.synthesize_verdict",
            # Free-form text with no VEREDICTO: pattern → _extract_verdict_from_llm_output returns None
            new=_make_synth_mock("No tengo suficiente contexto para evaluar esta afirmación de forma precisa."),
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
    # RAG verdict (FAKE) must survive: llm_verdict was None, no override happened
    assert "FAKE" in text
    assert "🔴" in text


# ── Test 7: MongoDB failure → Telegram reply still delivered ─────────────────

@pytest.mark.asyncio
async def test_pipeline_mongodb_failure_still_replies_telegram():
    """An insert_one exception must be swallowed so the verdict reply still reaches the user."""
    from services.worker_nlp.app.worker import process

    task = _make_task()
    message = FakeMessage(task)

    qdrant_hit = [
        {
            "score": 0.85,
            "verdict": "FAKE",
            "text": "Es falso.",
            "url": "https://maldita.es/test",
        }
    ]

    with (
        patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "fake_token"}),
        patch("services.worker_nlp.app.worker.is_gibberish", return_value=False),
        patch("services.worker_nlp.app.worker.get_cached_verdict", new=AsyncMock(return_value=None)),
        patch("services.worker_nlp.app.worker.extract_entities", return_value=["Gobierno", "virus"]),
        patch("services.worker_nlp.app.rag._search_qdrant", new=AsyncMock(return_value=qdrant_hit)),
        patch(
            "services.worker_nlp.app.worker.synthesize_verdict",
            new=_make_synth_mock("VEREDICTO: FALSO — confirmado por las fuentes."),
        ),
        patch("services.worker_nlp.app.worker.set_cached_verdict", new=AsyncMock()),
        patch("services.worker_nlp.app.worker.get_mongo_db", return_value=_make_mongo_mock_failing()),
        patch("services.worker_nlp.app.worker.Bot") as MockBot,
    ):
        mock_send = AsyncMock()
        MockBot.return_value.send_message = mock_send
        await process(message)  # must not raise despite MongoDB failure

    mock_send.assert_called_once()
    text = _sent_text(mock_send)
    assert "🔴" in text
    assert "FAKE" in text


# ── Test 8: URLs + emojis only → graceful UNVERIFIED degradation ──────────────

@pytest.mark.asyncio
async def test_pipeline_urls_emojis_only_degrades_to_unverified():
    """Text made up solely of URLs and emojis yields no entities;
    the pipeline must degrade gracefully and return UNVERIFIED with a useful message."""
    from services.worker_nlp.app.worker import process

    task = _make_task(text="\U0001f389\U0001f389 https://fake.com https://otro.com \U0001f4a5\U0001f4a5\U0001f4a5 !!!")
    message = FakeMessage(task)

    with (
        patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "fake_token"}),
        patch("services.worker_nlp.app.worker.is_gibberish", return_value=False),
        patch("services.worker_nlp.app.worker.get_cached_verdict", new=AsyncMock(return_value=None)),
        # NER strips URLs and emojis → no named entities remain
        patch("services.worker_nlp.app.worker.extract_entities", return_value=[]),
        # Defensive mocks in case hybrid_search doesn't early-exit
        patch("services.worker_nlp.app.rag._search_qdrant", new=AsyncMock(return_value=[])),
        patch("services.worker_nlp.app.rag._search_google_fact_check", new=AsyncMock(return_value=[])),
        patch("services.worker_nlp.app.rag._search_gnews", new=AsyncMock(return_value=[])),
        patch(
            "services.worker_nlp.app.worker.synthesize_verdict",
            new=_make_synth_mock("No tengo información sobre esta afirmación."),
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
    assert "UNVERIFIED" in text or "No encontré" in text


# ── Test 9: L2 off-topic hit → source URL suppressed, manual check suggested ─

@pytest.mark.asyncio
async def test_pipeline_l2_topic_mismatch_drops_source():
    """L2 returns an article with no entity overlap → source_url must be None
    and the Telegram reply must NOT contain 'https://' but MUST mention maldita.es."""
    from services.worker_nlp.app.worker import process

    task = _make_task(
        text=(
            "El virus del Ébola se está extendiendo por Europa según fuentes anónimas "
            "en redes sociales. Miles de personas podrían estar en peligro."
        )
    )
    message = FakeMessage(task)

    # L1 returns nothing; L2 Google FC returns a football article (zero entity overlap).
    # NOTE: the article must NOT contain any of the test entities ("Ébola", "Europa",
    # "virus") otherwise _topic_overlap_score would be > 0.25 and the hit would NOT
    # be filtered.  Removing "Copa de Europa" avoids the accidental "Europa" match.
    off_topic_hit = [
        {
            "score": 0.60,
            "verdict": "UNVERIFIED",
            "text": "El Real Madrid ganó la decimoquinta Copa en Wembley.",
            "url": "https://marca.com/futbol/real-madrid/copa-europa",
        }
    ]

    with (
        patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "fake_token", "NLP_TOPIC_OVERLAP_MIN": "0.25"}),
        patch("services.worker_nlp.app.worker.is_gibberish", return_value=False),
        patch("services.worker_nlp.app.worker.get_cached_verdict", new=AsyncMock(return_value=None)),
        patch("services.worker_nlp.app.worker.extract_entities", return_value=["Ébola", "Europa", "virus"]),
        patch("services.worker_nlp.app.rag._search_qdrant", new=AsyncMock(return_value=[])),
        patch("services.worker_nlp.app.rag._search_google_fact_check", new=AsyncMock(return_value=off_topic_hit)),
        patch("services.worker_nlp.app.rag._search_gnews", new=AsyncMock(return_value=[])),
        patch(
            "services.worker_nlp.app.worker.synthesize_verdict",
            # "NO VERIFICADO" → _extract_verdict_from_llm_output returns UNVERIFIED.
            # Summary must NOT contain no-info phrases so the Telegram formatter
            # takes the UNVERIFIED+no-source branch that appends maldita.es.
            # (Old tests used "no hay información relevante" which triggered the
            # Telegram _is_no_info_response check and bypassed the source-line.)
            new=_make_synth_mock("VEREDICTO: NO VERIFICADO — el artículo recuperado trata sobre fútbol y no guarda relación con la afirmación evaluada."),
        ),
        patch("services.worker_nlp.app.worker.set_cached_verdict", new=AsyncMock()),
        patch("services.worker_nlp.app.worker.get_mongo_db", return_value=_make_mongo_mock()),
        patch("services.worker_nlp.app.worker.Bot") as MockBot,
    ):
        mock_send = AsyncMock()
        MockBot.return_value.send_message = mock_send
        await process(message)

    mock_send.assert_called_once()
    reply_text = _sent_text(mock_send)
    assert "https://" not in reply_text, "Off-topic source URL must not appear in the reply"
    assert "maldita.es" in reply_text, "Manual verification suggestion must appear"
