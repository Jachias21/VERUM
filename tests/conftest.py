"""
Fixtures de pytest compartidas para todos los módulos de test.
"""
from __future__ import annotations

import uuid
import datetime
import pytest

from shared.schemas import NLPResult, TextTask


#  Modelo SpaCy (scope de sesión para evitar recarga en cada test) 

@pytest.fixture(scope="session")
def spacy_model():
    """Carga es_core_news_lg una sola vez por sesión de test."""
    try:
        import spacy
        return spacy.load("es_core_news_lg")
    except Exception:
        return None


#  Fixtures de dominio 

@pytest.fixture
def sample_text_task() -> TextTask:
    """TextTask realista con texto suficientemente largo para superar NLP_MIN_TEXT_LENGTH."""
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
    """NLPResult con retrieved_context y summary rellenos."""
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
    """Lista simulada de hits de Qdrant con los campos usados por hybrid_search."""
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
