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

    # A truly neutral statement that matches none of the verdict patterns
    assert _extract_verdict_from_llm_output("El artículo discute varios aspectos del tema.") is None


def test_extract_verdict_no_verified_before_verified():
    from services.worker_nlp.app.rag import _extract_verdict_from_llm_output

    assert _extract_verdict_from_llm_output(
        "VEREDICTO: NO VERIFICADO por falta de fuentes verificables"
    ) == "UNVERIFIED"


# ── New tests: Tier-2 natural-language verdict extraction ─────────────────────

def test_extract_verdict_natural_language_fake():
    from services.worker_nlp.app.rag import _extract_verdict_from_llm_output

    assert _extract_verdict_from_llm_output(
        "Después de analizar el artículo, puedo concluir que el mensaje viral es falso."
    ) == "FAKE"


def test_extract_verdict_natural_language_desmentido():
    from services.worker_nlp.app.rag import _extract_verdict_from_llm_output

    assert _extract_verdict_from_llm_output(
        "Esta afirmación ha sido desmentida por la CNMC."
    ) == "FAKE"


def test_extract_verdict_natural_language_unverified():
    from services.worker_nlp.app.rag import _extract_verdict_from_llm_output

    assert _extract_verdict_from_llm_output(
        "No hay suficiente información en el artículo para confirmar la afirmación."
    ) == "UNVERIFIED"


def test_extract_verdict_strict_wins_over_natural():
    """Tier-1 strict prefix must win even when NL signals suggest a different verdict."""
    from services.worker_nlp.app.rag import _extract_verdict_from_llm_output

    assert _extract_verdict_from_llm_output(
        "VEREDICTO: NO VERIFICADO — aunque el texto parece falso, no hay datos."
    ) == "UNVERIFIED"


# ── Tests: _topic_overlap_score ───────────────────────────────────────────────

def test_topic_overlap_score_zero_match():
    from services.worker_nlp.app.rag import _topic_overlap_score

    score = _topic_overlap_score(
        ["Ébola", "virus", "pandemia"],
        "El Real Madrid ganó la Copa de Europa en Wembley.",
    )
    assert score == 0.0


def test_topic_overlap_score_full_match():
    from services.worker_nlp.app.rag import _topic_overlap_score

    score = _topic_overlap_score(
        ["Madrid", "virus"],
        "El brote de virus en Madrid fue confirmado por las autoridades sanitarias.",
    )
    assert score == 1.0


def test_topic_overlap_score_accent_insensitive():
    """Accented entity must match its unaccented form in the article."""
    from services.worker_nlp.app.rag import _topic_overlap_score

    score = _topic_overlap_score(
        ["Ébola"],
        "Los expertos hablan sobre el ebola en Africa.",
    )
    assert score == 1.0


def test_topic_overlap_score_empty_entities():
    from services.worker_nlp.app.rag import _topic_overlap_score

    assert _topic_overlap_score([], "any article text here") == 0.0


# ── Test 10: hybrid_search with high-confidence Qdrant hit ───────────────────

async def test_hybrid_search_qdrant_hit(mock_qdrant_hits):
    fastembed = pytest.importorskip("fastembed", reason="fastembed not installed")
    import numpy as np
    SparseEmbedding = fastembed.SparseEmbedding

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
    # verdict is placeholder; final verdict is set by synthesize_verdict
    assert result.verdict == "UNVERIFIED"
    assert result.retrieved_context != ""
    assert result.summary == "", "summary must be empty until synthesize_verdict is called"


# ── Test 11: hybrid_search falls back to Google when Qdrant score is low ─────

async def test_hybrid_search_google_fallback():
    fastembed = pytest.importorskip("fastembed", reason="fastembed not installed")
    import numpy as np
    SparseEmbedding = fastembed.SparseEmbedding

    low_score_hit = [{"score": 0.30, "url": "https://qdrant.example.com", "verdict": "UNVERIFIED", "text": ""}]
    # text must contain "Madrid" so that _topic_overlap_score(["Madrid"], text) >= 0.25
    # (otherwise the L2 hit is filtered and verdict becomes UNVERIFIED).
    google_hits = [{"score": 1.0, "url": "https://factcheck.google.com/claim/1", "verdict": "FAKE", "text": "Fact checked article text about Madrid."}]

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

    # verdict is placeholder; final verdict is set by synthesize_verdict
    assert result.verdict == "UNVERIFIED"
    assert result.source_url == "https://factcheck.google.com/claim/1"
    assert "Fact checked article text" in result.retrieved_context


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

        await synthesize_verdict(result, "texto del mensaje viral")

    assert "context" in captured_kwargs, "LLM chain must receive 'context'"
    assert len(captured_kwargs["context"]) <= 1500, (
        f"context passed to LLM must be <= 1500 chars, got {len(captured_kwargs['context'])}"
    )


# ── Tests 14-18: _search_gnews and L2 parallel fallback ──────────────────────

async def test_search_gnews_returns_articles():
    """_search_gnews parses GNews JSON response and returns list with required fields."""
    articles_payload = {
        "articles": [
            {
                "title": "El bulo del 5G es falso",
                "description": "Expertos desmienten el bulo sobre el 5G.",
                "url": "https://gnews.example.com/article/1",
            },
            {
                "title": "Vacunas: sin evidencia de daño",
                "description": "No hay evidencia científica de efectos negativos.",
                "url": "https://gnews.example.com/article/2",
            },
        ]
    }

    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = articles_payload

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)

    with (
        patch("services.worker_nlp.app.rag.httpx.AsyncClient", return_value=mock_client),
        patch.dict("os.environ", {"GNEWS_API_KEY": "test_gnews_key"}),
    ):
        from services.worker_nlp.app.rag import _search_gnews

        results = await _search_gnews("bulo Madrid")

    assert len(results) == 2
    for r in results:
        assert "score" in r
        assert "verdict" in r
        assert "url" in r
        assert "text" in r
    assert results[0]["url"] == "https://gnews.example.com/article/1"
    assert results[1]["url"] == "https://gnews.example.com/article/2"


async def test_search_gnews_no_api_key_returns_empty():
    """_search_gnews returns [] immediately when GNEWS_API_KEY is not set."""
    with patch.dict("os.environ", {"GNEWS_API_KEY": ""}):
        from services.worker_nlp.app.rag import _search_gnews

        results = await _search_gnews("bulo Madrid")

    assert results == []


async def test_search_gnews_request_failure_returns_empty():
    """_search_gnews returns [] and does not propagate exceptions on request failure."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=Exception("Connection timeout"))

    with (
        patch("services.worker_nlp.app.rag.httpx.AsyncClient", return_value=mock_client),
        patch.dict("os.environ", {"GNEWS_API_KEY": "test_gnews_key"}),
    ):
        from services.worker_nlp.app.rag import _search_gnews

        results = await _search_gnews("bulo Madrid")

    assert results == []


async def test_hybrid_search_uses_gnews_when_google_fc_empty():
    """hybrid_search falls back to GNews when L1 score is low and Google FC returns nothing."""
    low_score_hits = [
        {"score": 0.30, "url": "https://qdrant.example.com", "verdict": "UNVERIFIED", "text": ""}
    ]
    gnews_hit = [
        {
            "score": 0.7,
            "verdict": "FAKE",
            "url": "https://gnews.example.com/article/gnews1",
            "text": "Este bulo sobre el virus en Madrid ha sido desmentido por múltiples fuentes.",
        }
    ]

    with (
        patch("services.worker_nlp.app.rag._search_qdrant", new=AsyncMock(return_value=low_score_hits)),
        patch("services.worker_nlp.app.rag._search_google_fact_check", new=AsyncMock(return_value=[])),
        patch("services.worker_nlp.app.rag._search_gnews", new=AsyncMock(return_value=gnews_hit)),
    ):
        from services.worker_nlp.app.rag import hybrid_search

        result = await hybrid_search(uuid.uuid4(), "Noticia de prueba sobre un bulo", ["Madrid", "virus"])

    # verdict is placeholder; final verdict is set by synthesize_verdict
    assert result.verdict == "UNVERIFIED"
    assert result.source_url == "https://gnews.example.com/article/gnews1"
    assert result.retrieved_context == "Este bulo sobre el virus en Madrid ha sido desmentido por múltiples fuentes."
    assert result.fact_check_matches == 1


async def test_hybrid_search_prioritizes_google_fc_over_gnews():
    """hybrid_search uses Google FC result, not GNews, when both L2 sources return hits."""
    low_score_hits = [
        {"score": 0.30, "url": "https://qdrant.example.com", "verdict": "UNVERIFIED", "text": ""}
    ]
    google_hit = [
        {
            "score": 1.0,
            "verdict": "REAL",
            "url": "https://factcheck.google.com/claim/99",
            "text": "La afirmación sobre Madrid es verdadera según múltiples fuentes verificadas.",
        }
    ]
    gnews_hit = [
        {
            "score": 0.7,
            "verdict": "FAKE",
            "url": "https://gnews.example.com/article/99",
            "text": "Artículo de GNews sobre Madrid con veredicto diferente.",
        }
    ]

    with (
        patch("services.worker_nlp.app.rag._search_qdrant", new=AsyncMock(return_value=low_score_hits)),
        patch("services.worker_nlp.app.rag._search_google_fact_check", new=AsyncMock(return_value=google_hit)),
        patch("services.worker_nlp.app.rag._search_gnews", new=AsyncMock(return_value=gnews_hit)),
    ):
        from services.worker_nlp.app.rag import hybrid_search

        result = await hybrid_search(uuid.uuid4(), "Noticia de prueba", ["Madrid"])

    # verdict is placeholder; final verdict is set by synthesize_verdict
    assert result.verdict == "UNVERIFIED"
    assert result.source_url == "https://factcheck.google.com/claim/99"
    assert result.retrieved_context == "La afirmación sobre Madrid es verdadera según múltiples fuentes verificadas."


# ── Test: L3 sentinel — no context available, LLM uses general knowledge ──────

async def test_hybrid_search_l3_no_context_uses_general_knowledge_sentinel():
    """When no source provides reliable context, hybrid_search returns the L3 sentinel."""
    from services.worker_nlp.app.rag import hybrid_search

    with (
        patch("services.worker_nlp.app.rag._search_qdrant", new=AsyncMock(return_value=[])),
        patch("services.worker_nlp.app.rag._search_google_fact_check", new=AsyncMock(return_value=[])),
        patch("services.worker_nlp.app.rag._search_gnews", new=AsyncMock(return_value=[])),
    ):
        result = await hybrid_search(
            uuid.uuid4(),
            "El Muro de Berlín cayó en 1989, es un hecho histórico bien documentado.",
            ["Muro de Berlín", "1989"],
        )

    assert result.summary == "__USE_GENERAL_KNOWLEDGE__"
    assert result.retrieved_context == ""
    assert result.fact_check_matches == 0
