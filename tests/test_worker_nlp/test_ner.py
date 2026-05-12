"""
Tests for services/worker_nlp/app/ner.py
"""
from __future__ import annotations

import pytest

try:
    import spacy
    spacy.load("es_core_news_lg")
    _SPACY_AVAILABLE = True
except Exception:
    _SPACY_AVAILABLE = False

spacy_required = pytest.mark.skipif(
    not _SPACY_AVAILABLE,
    reason="es_core_news_lg not installed",
)


# ── Test 1: extract_entities with known text ─────────────────────────────────

@spacy_required
def test_extract_entities_known_text():
    from services.worker_nlp.app.ner import extract_entities

    text = "El hacker de Madrid robó datos de la Policía Nacional"
    entities = extract_entities(text)

    assert isinstance(entities, list)
    # At least both key entities must be present
    joined = " ".join(entities).lower()
    assert "madrid" in joined, f"Expected 'Madrid' in entities, got: {entities}"
    assert any("polic" in e.lower() for e in entities), (
        f"Expected 'Policía Nacional' in entities, got: {entities}"
    )


# ── Test 2: extract_entities with empty text ─────────────────────────────────

@spacy_required
def test_extract_entities_empty_text():
    from services.worker_nlp.app.ner import extract_entities

    assert extract_entities("") == []


# ── Test 3: is_gibberish with random chars ───────────────────────────────────

@spacy_required
def test_is_gibberish_random_chars():
    from services.worker_nlp.app.ner import is_gibberish

    assert is_gibberish("asdf qwer zxcv 1234 !!") is True


# ── Test 4: is_gibberish with normal Spanish text ────────────────────────────

@spacy_required
def test_is_gibberish_normal_text():
    from services.worker_nlp.app.ner import is_gibberish

    text = (
        "El Gobierno de España anuncia nuevas medidas para proteger "
        "la salud pública ante la llegada del otoño."
    )
    assert len(text) >= 60
    assert is_gibberish(text) is False


# ── Test 5: is_gibberish with very short text ────────────────────────────────

@spacy_required
def test_is_gibberish_very_short_text():
    from services.worker_nlp.app.ner import is_gibberish

    assert is_gibberish("hi") is True
