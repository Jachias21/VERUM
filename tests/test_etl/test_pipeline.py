"""
Tests for services/etl/app/pipeline.py
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest


# ── Test 18: _infer_verdict_from_entry — "falso" in title → FAKE ─────────────

def test_infer_verdict_fake_keyword_in_title():
    from services.etl.app.pipeline import _infer_verdict_from_entry

    entry = {"title": "Es falso que el 5G cause cáncer", "summary": "", "tags": []}
    assert _infer_verdict_from_entry(entry, publisher="Reuters") == "FAKE"


# ── Test 19: _infer_verdict_from_entry — fact-checker publisher → FAKE ────────

def test_infer_verdict_factchecker_publisher_heuristic():
    from services.etl.app.pipeline import _infer_verdict_from_entry

    # No FAKE/REAL keywords in the content — falls through to publisher heuristic
    entry = {"title": "Análisis de la semana en redes sociales", "summary": "Resumen semanal sin palabras clave.", "tags": []}
    assert _infer_verdict_from_entry(entry, publisher="maldita") == "FAKE"


# ── Test 20: _infer_verdict_from_entry — neutral entry → UNVERIFIED ──────────

def test_infer_verdict_neutral_entry():
    from services.etl.app.pipeline import _infer_verdict_from_entry

    entry = {"title": "Noticias del día en España", "summary": "El tiempo en Madrid es soleado.", "tags": []}
    assert _infer_verdict_from_entry(entry, publisher="Generic News") == "UNVERIFIED"


# ── Test 21: _validate_article — valid article → True ────────────────────────

def test_validate_article_valid():
    from services.etl.app.pipeline import _validate_article

    article = {
        "title": "Este es un título válido de diez chars o más",
        "summary": "Este es un resumen suficientemente largo para pasar la validación.",
        "url": "https://example.com/article/123",
    }
    assert _validate_article(article) is True


# ── Test 22: _validate_article — empty url → False ───────────────────────────

def test_validate_article_empty_url():
    from services.etl.app.pipeline import _validate_article

    article = {
        "title": "Título válido de diez caracteres",
        "summary": "Resumen suficientemente largo para la validación del artículo.",
        "url": "",
    }
    assert _validate_article(article) is False


# ── Test 23: _validate_article — short title → False ────────────────────────

def test_validate_article_short_title():
    from services.etl.app.pipeline import _validate_article

    article = {
        "title": "Corto",          # 5 chars < 10
        "summary": "Resumen suficientemente largo para la validación del artículo.",
        "url": "https://example.com/article/456",
    }
    assert _validate_article(article) is False


# ── Test 24a: _ensure_collection_schema — collection absent → create_collection called ──

def test_ensure_collection_schema_creates_when_absent():
    from services.etl.app.pipeline import _ensure_collection_schema

    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []  # no existing collections

    _ensure_collection_schema(mock_client, "fact_checks")

    mock_client.create_collection.assert_called_once()
    mock_client.delete_collection.assert_not_called()


# ── Test 24b: _ensure_collection_schema — collection present → no drop ───────

def test_ensure_collection_schema_no_drop_when_present():
    from services.etl.app.pipeline import _ensure_collection_schema

    existing = MagicMock()
    existing.name = "fact_checks"
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = [existing]

    _ensure_collection_schema(mock_client, "fact_checks")

    mock_client.delete_collection.assert_not_called()
    mock_client.create_collection.assert_not_called()


# ── Test 24c: _ensure_collection_schema — force_recreate drops and recreates ──

def test_ensure_collection_schema_force_recreate():
    from services.etl.app.pipeline import _ensure_collection_schema

    existing = MagicMock()
    existing.name = "fact_checks"
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = [existing]

    _ensure_collection_schema(mock_client, "fact_checks", force_recreate=True)

    mock_client.delete_collection.assert_called_once_with("fact_checks")
    mock_client.create_collection.assert_called_once()
