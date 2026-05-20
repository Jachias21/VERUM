"""
Tests for services/worker_nlp/app/ner.py
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ── SpaCy availability guards ─────────────────────────────────────────────────

# Tier 1: the spacy *package* is importable (sufficient for mock-based tests)
try:
    import spacy as _spacy
    _SPACY_IMPORTABLE = True
except ImportError:
    _SPACY_IMPORTABLE = False

# Tier 2: the Spanish model is actually loadable (needed for real-SpaCy tests)
_SPACY_AVAILABLE = False
if _SPACY_IMPORTABLE:
    try:
        _spacy.load("es_core_news_lg")
        _SPACY_AVAILABLE = True
    except Exception:
        pass

# Tier 3: the English model is loadable (needed for the English NER test)
_EN_SPACY_AVAILABLE = False
if _SPACY_IMPORTABLE:
    try:
        _spacy.load("en_core_web_sm")
        _EN_SPACY_AVAILABLE = True
    except Exception:
        pass

# Mark: spacy package importable (module-level import of ner.py succeeds)
spacy_importable = pytest.mark.skipif(
    not _SPACY_IMPORTABLE,
    reason="spacy package not installed",
)

# Mark: es_core_news_lg model available (real SpaCy inference)
spacy_required = pytest.mark.skipif(
    not _SPACY_AVAILABLE,
    reason="es_core_news_lg not installed",
)

# Mark: en_core_web_sm model available
en_spacy_required = pytest.mark.skipif(
    not _EN_SPACY_AVAILABLE,
    reason="en_core_web_sm not installed",
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


# ── Test 6: is_gibberish with long random tokens (no vowels) ────────────────

@spacy_required
def test_is_gibberish_long_random_tokens():
    from services.worker_nlp.app.ner import is_gibberish

    assert is_gibberish(
        "asdf qwer zxcv 1234 ñlkj mnbv poiu hgfd asdrrg efef pqpwpwpe"
    ) is True


# ── Test 7: extract_entities with English text → uses en_core_web_sm ─────────

@spacy_required
@en_spacy_required
def test_extract_entities_english_uses_en_model():
    from services.worker_nlp.app.ner import extract_entities

    # Force language detection to return "en" to ensure the English model is used
    with patch("services.worker_nlp.app.ner._detect_language", return_value="en"):
        entities = extract_entities(
            "Apple and Microsoft are both headquartered in the United States of America"
        )

    assert isinstance(entities, list)
    assert len(entities) > 0


# ── Test 7: extract_entities falls back to noun chunks when NER yields nothing ─

@spacy_importable
def test_extract_entities_noun_chunks_fallback():
    """When the SpaCy doc has no named entities, noun chunks must be used."""
    from services.worker_nlp.app.ner import extract_entities

    mock_chunk1 = MagicMock()
    mock_chunk1.text = "el texto extraño"
    mock_chunk1.root.is_stop = False
    mock_chunk2 = MagicMock()
    mock_chunk2.text = "las palabras raras"
    mock_chunk2.root.is_stop = False

    mock_doc = MagicMock()
    mock_doc.ents = []  # No named entities
    mock_doc.noun_chunks = [mock_chunk1, mock_chunk2]

    mock_nlp = MagicMock(return_value=mock_doc)

    with patch("services.worker_nlp.app.ner._get_nlp", return_value=mock_nlp), \
         patch("services.worker_nlp.app.ner._detect_language", return_value="es"):
        entities = extract_entities("some text without named entities here")

    assert "el texto extraño" in entities
    assert "las palabras raras" in entities


# ── Test 8: extract_entities strips URLs and emojis before NLP processing ──────

@spacy_importable
def test_extract_entities_cleans_urls_and_emojis():
    """URLs and emoji characters must be stripped from the text before SpaCy runs."""
    from services.worker_nlp.app.ner import extract_entities

    mock_doc = MagicMock()
    mock_doc.ents = []
    mock_doc.noun_chunks = []

    mock_nlp = MagicMock(return_value=mock_doc)

    with patch("services.worker_nlp.app.ner._get_nlp", return_value=mock_nlp), \
         patch("services.worker_nlp.app.ner._detect_language", return_value="es"):
        extract_entities("Mira este bulo 🎉 https://fake-news.com/article 💥 !!!")

    # The text actually passed into the SpaCy pipeline must have no URL or emoji
    called_text = mock_nlp.call_args[0][0]
    assert "https" not in called_text
    assert "fake-news.com" not in called_text
    assert "🎉" not in called_text
    assert "💥" not in called_text


# ── Test 9: extract_entities returns a deduplicated list ──────────────────────

@spacy_importable
def test_extract_entities_deduplicates():
    """The same entity string appearing multiple times must appear only once."""
    from services.worker_nlp.app.ner import extract_entities

    mock_ent1 = MagicMock()
    mock_ent1.text = "Pedro Sánchez"
    mock_ent2 = MagicMock()
    mock_ent2.text = "Pedro Sánchez"  # exact duplicate

    mock_doc = MagicMock()
    mock_doc.ents = [mock_ent1, mock_ent2]
    mock_doc.noun_chunks = []

    mock_nlp = MagicMock(return_value=mock_doc)

    with patch("services.worker_nlp.app.ner._get_nlp", return_value=mock_nlp), \
         patch("services.worker_nlp.app.ner._detect_language", return_value="es"):
        entities = extract_entities("Pedro Sánchez habló sobre Pedro Sánchez")

    assert entities.count("Pedro Sánchez") == 1


# ── Test 10: extract_entities filters entities with ≤2 characters ─────────────

@spacy_importable
def test_extract_entities_filters_short_strings():
    """Entities whose stripped text is 2 characters or fewer must be discarded."""
    from services.worker_nlp.app.ner import extract_entities

    mock_ent_short = MagicMock()
    mock_ent_short.text = "EU"       # 2 chars — must be filtered (len > 2 required)
    mock_ent_valid = MagicMock()
    mock_ent_valid.text = "Unión Europea"  # 13 chars — must be kept

    mock_doc = MagicMock()
    mock_doc.ents = [mock_ent_short, mock_ent_valid]
    mock_doc.noun_chunks = []

    mock_nlp = MagicMock(return_value=mock_doc)

    with patch("services.worker_nlp.app.ner._get_nlp", return_value=mock_nlp), \
         patch("services.worker_nlp.app.ner._detect_language", return_value="es"):
        entities = extract_entities("La EU forma parte de la Unión Europea")

    assert "EU" not in entities
    assert "Unión Europea" in entities


# ── Test 11: _detect_language with Spanish text → "es" ───────────────────────

@spacy_importable
def test_detect_language_returns_es_for_spanish():
    pytest.importorskip("langdetect")
    from services.worker_nlp.app.ner import _detect_language

    with patch("langdetect.detect", return_value="es"):
        result = _detect_language(
            "El Gobierno de España anuncia nuevas medidas de salud pública."
        )

    assert result == "es"


# ── Test 12: _detect_language with English text → "en" ───────────────────────

@spacy_importable
def test_detect_language_returns_en_for_english():
    pytest.importorskip("langdetect")
    from services.worker_nlp.app.ner import _detect_language

    with patch("langdetect.detect", return_value="en"):
        result = _detect_language(
            "The government of Spain announces new public health measures."
        )

    assert result == "en"
