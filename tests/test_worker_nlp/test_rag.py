"""
Tests for services/worker_nlp/app/rag.py
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.schemas import NLPResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_result(**kwargs) -> NLPResult:
    defaults = dict(
        query_id=uuid.uuid4(),
        extracted_entities=[],
        fact_check_matches=0,
        verdict="UNVERIFIED",
        retrieved_context="",
        summary="",
    )
    defaults.update(kwargs)
    return NLPResult(**defaults)


# ── Tests 6-9: _extract_verdict_from_llm_output ──────────────────────────────

def test_extract_verdict_fake():
    from services.worker_nlp.app.rag import _extract_verdict_from_llm_output

    assert _extract_verdict_from_llm_output("VEREDICTO: FALSO esto es un bulo") == "FAKE"


def test_extract_verdict_real():
    from services.worker_nlp.app.rag import _extract_verdict_from_llm_output

    assert _extract_verdict_from_llm_output("VEREDICTO: VERDADERO la noticia es real") == "REAL"


def test_extract_verdict_unverified():
    from services.worker_nlp.app.rag import _extract_verdict_from_llm_output

    assert _extract_verdict_from_llm_output("VEREDICTO: NO VERIFICADO") == "UNVERIFIED"


def test_extract_verdict_no_match():
    from services.worker_nlp.app.rag import _extract_verdict_from_llm_output

    assert _extract_verdict_from_llm_output("No hay suficiente información disponible.") is None


# ── Test 10: hybrid_search with high-confidence Qdrant hit ───────────────────

async def test_hybrid_search_qdrant_hit(mock_qdrant_hits):
    import numpy as np
    from fastembed import SparseEmbedding

    fake_dense = [0.1] * 1024
    fake_sparse = SparseEmbedding(
        indices=np.array([0, 1, 2]),
        values=np.array([0.5, 0.3, 0.2]),
    )

    with (
        patch("services.worker_nlp.app.rag._search_qdrant", new=AsyncMock(return_value=mock_qdrant_hits)),
        patch("services.worker_nlp.app.rag.embed", return_value=[fake_dense]),
        patch("services.worker_nlp.app.rag.sparse_embed", return_value=[fake_sparse]),
    ):
        from services.worker_nlp.app.rag import hybrid_search

        qid = uuid.uuid4()
        result = await hybrid_search(qid, "El virus de Madrid no existe", ["Madrid", "virus"])

    assert isinstance(result, NLPResult)
    assert result.verdict == "FAKE"
    assert result.retrieved_context != ""
    assert result.summary == "", "summary must be empty until synthesize_verdict is called"


# ── Test 11: hybrid_search falls back to Google when Qdrant score is low ─────

async def test_hybrid_search_google_fallback():
    import numpy as np
    from fastembed import SparseEmbedding

    low_score_hit = [{"score": 0.30, "url": "https://qdrant.example.com", "verdict": "UNVERIFIED", "text": ""}]
    google_hits = [{"score": 1.0, "url": "https://factcheck.google.com/claim/1", "verdict": "FAKE", "text": "Fact checked article text."}]

    fake_dense = [0.1] * 1024
    fake_sparse = SparseEmbedding(
        indices=np.array([0]),
        values=np.array([1.0]),
    )

    with (
        patch("services.worker_nlp.app.rag._search_qdrant", new=AsyncMock(return_value=low_score_hit)),
        patch("services.worker_nlp.app.rag._search_google_fact_check", new=AsyncMock(return_value=google_hits)),
        patch("services.worker_nlp.app.rag._search_gnews", new=AsyncMock(return_value=[])),
        patch("services.worker_nlp.app.rag.embed", return_value=[fake_dense]),
        patch("services.worker_nlp.app.rag.sparse_embed", return_value=[fake_sparse]),
    ):
        from services.worker_nlp.app.rag import hybrid_search

        result = await hybrid_search(uuid.uuid4(), "Noticia de prueba para test", ["Madrid"])

    assert result.verdict == "FAKE"
    assert result.source_url == "https://factcheck.google.com/claim/1"
    assert result.retrieved_context == "Fact checked article text."


# ── Test 12: hybrid_search early exit — text too short and no entities ────────

async def test_hybrid_search_early_exit_short_text():
    from services.worker_nlp.app.rag import hybrid_search

    result = await hybrid_search(uuid.uuid4(), "corto", [])

    assert result.verdict == "UNVERIFIED"
    assert result.summary != ""   # user-facing message must be present
    assert result.retrieved_context == ""


# ── Test 13: synthesize_verdict truncates retrieved_context to 1500 chars ────

async def test_synthesize_verdict_truncates_context():
    long_context = "A" * 3000
    result = _make_result(retrieved_context=long_context, verdict="FAKE")

    captured_kwargs: dict = {}

    async def fake_ainvoke(inputs: dict) -> str:
        captured_kwargs.update(inputs)
        return "VEREDICTO: FALSO texto truncado correctamente."

    mock_chain = MagicMock()
    mock_chain.ainvoke = fake_ainvoke

    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)

    with (
        patch("services.worker_nlp.app.rag.PromptTemplate", return_value=mock_prompt),
        patch("services.worker_nlp.app.rag.OllamaLLM", return_value=MagicMock()),
        patch.dict("os.environ", {"OLLAMA_HOST": "localhost", "OLLAMA_PORT": "11434", "OLLAMA_MODEL": "llama3.2:3b"}),
    ):
        from services.worker_nlp.app.rag import synthesize_verdict

        await synthesize_verdict("texto del mensaje viral", result)

    assert "retrieved_article" in captured_kwargs, "LLM chain must receive 'retrieved_article'"
    assert len(captured_kwargs["retrieved_article"]) <= 1500, (
        f"retrieved_article passed to LLM must be <= 1500 chars, got {len(captured_kwargs['retrieved_article'])}"
    )
