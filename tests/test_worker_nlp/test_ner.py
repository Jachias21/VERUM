"""
Tests para services/worker_nlp/app/ner.py
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

# Nivel 2: el modelo en español es cargable (necesario para tests con SpaCy real)
_SPACY_AVAILABLE = False
if _SPACY_IMPORTABLE:
    try:
        _spacy.load("es_core_news_lg")
        _SPACY_AVAILABLE = True
    except Exception:
        pass

# Nivel 3: el modelo en inglés es cargable (necesario para el test NER en inglés)
_EN_SPACY_AVAILABLE = False
if _SPACY_IMPORTABLE:
    try:
        _spacy.load("en_core_web_sm")
        _EN_SPACY_AVAILABLE = True
    except Exception:
        pass

# Mark: paquete spacy importable (la importación de ner.py tiene éxito)
spacy_importable = pytest.mark.skipif(
    not _SPACY_IMPORTABLE,
    reason="paquete spacy no instalado",
)

# Mark: modelo es_core_news_lg disponible (inferencia SpaCy real)
spacy_required = pytest.mark.skipif(
    not _SPACY_AVAILABLE,
    reason="es_core_news_lg no instalado",
)

# Mark: modelo en_core_web_sm disponible
en_spacy_required = pytest.mark.skipif(
    not _EN_SPACY_AVAILABLE,
    reason="en_core_web_sm no instalado",
)


# Test 1: extract_entities con texto conocido 
@spacy_required
def test_extract_entities_known_text():
    from services.worker_nlp.app.ner import extract_entities

    text = "El hacker de Madrid robó datos de la Policía Nacional"
    entities = extract_entities(text)

    assert isinstance(entities, list)
    # Al menos ambas entidades clave deben estar presentes
    joined = " ".join(entities).lower()
    assert "madrid" in joined, f"Se esperaba 'Madrid' en entidades, obtenido: {entities}"
    assert any("polic" in e.lower() for e in entities), (
        f"Se esperaba 'Policía Nacional' en entidades, obtenido: {entities}"
    )


#  Test 2: extract_entities con texto vacío 

@spacy_required
def test_extract_entities_empty_text():
    from services.worker_nlp.app.ner import extract_entities

    assert extract_entities("") == []


#  Test 3: is_gibberish con caracteres aleatorios  
@spacy_required
def test_is_gibberish_random_chars():
    from services.worker_nlp.app.ner import is_gibberish

    assert is_gibberish("asdf qwer zxcv 1234 !!") is True


#  Test 4: is_gibberish con texto español normal 

@spacy_required
def test_is_gibberish_normal_text():
    from services.worker_nlp.app.ner import is_gibberish

    text = (
        "El Gobierno de España anuncia nuevas medidas para proteger "
        "la salud pública ante la llegada del otoño."
    )
    assert len(text) >= 60
    assert is_gibberish(text) is False


#  Test 5: is_gibberish con texto muy corto 

@spacy_required
def test_is_gibberish_very_short_text():
    from services.worker_nlp.app.ner import is_gibberish

    assert is_gibberish("hi") is True


#  Test 6: is_gibberish con tokens largos aleatorios (sin vocales) 

@spacy_required
def test_is_gibberish_long_random_tokens():
    from services.worker_nlp.app.ner import is_gibberish

    assert is_gibberish(
        "asdf qwer zxcv 1234 ñlkj mnbv poiu hgfd asdrrg efef pqpwpwpe"
    ) is True


# ── Test 7: extract_entities con texto inglés → usa en_core_web_sm ──────────

@spacy_required
@en_spacy_required
def test_extract_entities_english_uses_en_model():
    from services.worker_nlp.app.ner import extract_entities

    # Forzar que la detección de idioma devuelva "en" para asegurar que se usa el modelo inglés
    with patch("services.worker_nlp.app.ner._detect_language", return_value="en"):
        entities = extract_entities(
            "Apple and Microsoft are both headquartered in the United States of America"
        )

    assert isinstance(entities, list)
    assert len(entities) > 0


# ── Test 7: extract_entities cae a noun chunks cuando NER no devuelve nada ───

@spacy_importable
def test_extract_entities_noun_chunks_fallback():
    """Cuando el doc SpaCy no tiene entidades nombradas, se deben usar noun chunks."""
    from services.worker_nlp.app.ner import extract_entities

    mock_chunk1 = MagicMock()
    mock_chunk1.text = "el texto extraño"
    mock_chunk1.root.is_stop = False
    mock_chunk2 = MagicMock()
    mock_chunk2.text = "las palabras raras"
    mock_chunk2.root.is_stop = False

    mock_doc = MagicMock()
    mock_doc.ents = []  # Sin entidades nombradas
    mock_doc.noun_chunks = [mock_chunk1, mock_chunk2]

    mock_nlp = MagicMock(return_value=mock_doc)

    with patch("services.worker_nlp.app.ner._get_nlp", return_value=mock_nlp), \
         patch("services.worker_nlp.app.ner._detect_language", return_value="es"):
        entities = extract_entities("some text without named entities here")

    assert "el texto extraño" in entities
    assert "las palabras raras" in entities


# ── Test 8: extract_entities limpia URLs y emojis antes del procesado NLP ────

@spacy_importable
def test_extract_entities_cleans_urls_and_emojis():
    """Las URLs y emojis deben eliminarse del texto antes de ejecutar SpaCy."""
    from services.worker_nlp.app.ner import extract_entities

    mock_doc = MagicMock()
    mock_doc.ents = []
    mock_doc.noun_chunks = []

    mock_nlp = MagicMock(return_value=mock_doc)

    with patch("services.worker_nlp.app.ner._get_nlp", return_value=mock_nlp), \
         patch("services.worker_nlp.app.ner._detect_language", return_value="es"):
        extract_entities("Mira este bulo 🎉 https://fake-news.com/article 💥 !!!")

    # El texto pasado al pipeline SpaCy no debe tener URL ni emoji
    called_text = mock_nlp.call_args[0][0]
    assert "https" not in called_text
    assert "fake-news.com" not in called_text
    assert "🎉" not in called_text
    assert "💥" not in called_text


# ── Test 9: extract_entities devuelve lista deduplicada ──────────────────────

@spacy_importable
def test_extract_entities_deduplicates():
    """La misma cadena de entidad que aparece varias veces debe aparecer solo una vez."""
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


# ── Test 10: is_gibberish with long invented words that contain vowels ────────

@spacy_required
def test_is_gibberish_long_invented_words_with_vowels():
    """Invented strings with vowels (OOV / consonant-cluster signals) → True."""
    from services.worker_nlp.app.ner import is_gibberish

    assert is_gibberish(
        "Jsjfkwnx iajdiqbeufjfjf jgisosoe ufan irjcnauwjdjf isow os irnwiend"
    ) is True


# ── Test 11: is_gibberish with pseudo-acronym mash ───────────────────────────

@spacy_required
def test_is_gibberish_pseudoacronym_mash():
    """Tokens dominated by consecutive consonant clusters → True."""
    from services.worker_nlp.app.ner import is_gibberish

    assert is_gibberish(
        "Plkmqaxsv brnkdjwm xklqpoiuy tjsfnvbcx mnhpkrwz"
    ) is True


# ── Test 12: real Spanish text must NOT be flagged as gibberish (regression) ──

@spacy_required
def test_is_gibberish_real_spanish_still_passes():
    """A well-formed Spanish sentence must not be flagged as gibberish."""
    from services.worker_nlp.app.ner import is_gibberish

    assert is_gibberish(
        "El presidente del Gobierno de España visitó Bruselas para reunirse con "
        "la Comisión Europea sobre los fondos de recuperación post-pandemia."
    ) is False


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
