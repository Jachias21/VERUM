"""
Shared pytest fixtures for all test modules.
"""
from __future__ import annotations

import uuid
import datetime
import pytest

from shared.schemas import NLPResult, TextTask


# ── SpaCy model (session-scoped to avoid reloading on each test) ──────────────

@pytest.fixture(scope="session")
def spacy_model():
    """Load es_core_news_lg once per test session."""
    try:
        import spacy
        return spacy.load("es_core_news_lg")
    except Exception:
        return None


# ── Domain fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def sample_text_task() -> TextTask:
    """A realistic TextTask with a text long enough to pass NLP_MIN_TEXT_LENGTH."""
    return TextTask(
        query_id=uuid.uuid4(),
        user_hash="a" * 64,
        chat_id=123456789,
        text=(
            "El Gobierno de España anuncia que el virus detectado en Madrid "
            "no supone un riesgo para la salud pública según los expertos."
        ),
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )


@pytest.fixture
def mock_nlp_result() -> NLPResult:
    """An NLPResult with both retrieved_context and summary populated."""
    return NLPResult(
        query_id=uuid.uuid4(),
        extracted_entities=["Madrid", "Gobierno de España"],
        fact_check_matches=2,
        source_url="https://example.com/article",
        verdict="FAKE",
        retrieved_context="Este es el texto del artículo fuente recuperado de Qdrant.",
        summary="VEREDICTO: FALSO — El artículo confirma que la noticia es un bulo.",
    )


@pytest.fixture
def mock_qdrant_hits() -> list[dict]:
    """Simulated Qdrant hit list with the fields used by hybrid_search."""
    return [
        {
            "score": 0.92,
            "url": "https://maldita.es/article/123",
            "verdict": "FAKE",
            "text": (
                "Este bulo sobre el virus circuló en redes sociales en 2023. "
                "Según los expertos, la información es falsa y carece de evidencia científica."
            ),
        }
    ]
