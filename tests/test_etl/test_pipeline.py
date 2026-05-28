"""
Tests for services/etl/app/pipeline.py
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("feedparser", reason="feedparser not installed (Docker-only dependency)")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_article():
    """Base valid article that passes _validate_article."""
    return {
        "title": "Este es un título válido de más de diez caracteres",
        "summary": "Este es un resumen suficientemente largo para pasar la validación.",
        "url": "https://example.com/article/123",
    }


# ── _validate_article ─────────────────────────────────────────────────────────

def test_validate_article_valid(valid_article):
    from services.etl.app.pipeline import _validate_article

    assert _validate_article(valid_article) is True


def test_validate_article_short_title(valid_article):
    from services.etl.app.pipeline import _validate_article

    valid_article["title"] = "Corto"  # 5 chars < 10
    assert _validate_article(valid_article) is False


def test_validate_article_short_summary(valid_article):
    from services.etl.app.pipeline import _validate_article

    valid_article["summary"] = "Muy corto"  # 9 chars < 20
    assert _validate_article(valid_article) is False


def test_validate_article_non_http_url(valid_article):
    from services.etl.app.pipeline import _validate_article

    valid_article["url"] = "ftp://example.com/article"  # does not start with "http"
    assert _validate_article(valid_article) is False


def test_validate_article_empty_url(valid_article):
    from services.etl.app.pipeline import _validate_article

    valid_article["url"] = ""
    assert _validate_article(valid_article) is False


# ── _infer_verdict_from_entry ─────────────────────────────────────────────────

def test_infer_verdict_fake_keyword_in_tags():
    from services.etl.app.pipeline import _infer_verdict_from_entry

    entry = {
        "title": "Análisis de contenido viral en redes sociales",
        "summary": "Revisión de publicaciones en redes sociales.",
        "tags": [{"term": "falso"}, {"term": "bulo"}],
    }
    assert _infer_verdict_from_entry(entry) == "FAKE"


def test_infer_verdict_fake_keyword_in_title():
    from services.etl.app.pipeline import _infer_verdict_from_entry

    entry = {"title": "Es falso que el 5G cause cáncer", "summary": "", "tags": []}
    assert _infer_verdict_from_entry(
        entry, source_url="https://reuters.com/article/123"
    ) == "FAKE"


def test_infer_verdict_real_keyword_verified():
    from services.etl.app.pipeline import _infer_verdict_from_entry

    entry = {
        "title": "Verified: vaccines are safe and effective according to studies",
        "summary": "Health authorities confirm the data after extensive research.",
        "tags": [],
    }
    assert _infer_verdict_from_entry(entry) == "REAL"


def test_infer_verdict_factchecker_domain_without_keywords_is_unverified():
    """Maldita.es URL without lexical signals → UNVERIFIED."""
    from services.etl.app.pipeline import _infer_verdict_from_entry

    entry = {
        "title": "Análisis de la semana en redes sociales",
        "summary": "Resumen semanal sin palabras clave.",
        "tags": [],
    }
    assert _infer_verdict_from_entry(
        entry, source_url="https://maldita.es/feed/"
    ) == "UNVERIFIED"


def test_infer_verdict_factchecker_domain_with_debunk_keywords_is_fake():
    """Lexical FAKE signal beats domain class."""
    from services.etl.app.pipeline import _infer_verdict_from_entry

    entry = {
        "title": "Es falso que las vacunas contengan microchips",
        "summary": "Maldita desmiente este bulo viral.",
        "tags": [],
    }
    assert _infer_verdict_from_entry(
        entry, source_url="https://maldita.es/feed/"
    ) == "FAKE"


def test_infer_verdict_institutional_domain_defaults_to_real():
    """Moncloa URL without lexical FAKE signals → REAL."""
    from services.etl.app.pipeline import _infer_verdict_from_entry

    entry = {
        "title": "El Gobierno aprueba el nuevo Real Decreto de pensiones",
        "summary": "El Consejo de Ministros aprueba la subida del SMI.",
        "tags": [],
    }
    assert _infer_verdict_from_entry(
        entry, source_url="https://www.lamoncloa.gob.es/some/path"
    ) == "REAL"


def test_infer_verdict_institutional_subdomain_defaults_to_real():
    """Subdomains of institutional domains must also be recognized."""
    from services.etl.app.pipeline import _infer_verdict_from_entry

    entry = {
        "title": "Latest from European Central Bank press release",
        "summary": "The ECB raises interest rates by 25 basis points.",
        "tags": [],
    }
    assert _infer_verdict_from_entry(
        entry, source_url="https://feeds.ecb.europa.eu/press/release/123"
    ) == "REAL"


def test_infer_verdict_institutional_domain_with_fake_keyword_stays_fake():
    """Lexical FAKE signal beats institutional default."""
    from services.etl.app.pipeline import _infer_verdict_from_entry

    entry = {
        "title": "La OMS desmiente el bulo sobre la vacuna",
        "summary": "Es falso lo que circula en redes.",
        "tags": [],
    }
    assert _infer_verdict_from_entry(
        entry, source_url="https://www.who.int/news/item/123"
    ) == "FAKE"


def test_infer_verdict_both_signals_is_unverified():
    """Both FAKE and REAL keywords present → UNVERIFIED."""
    from services.etl.app.pipeline import _infer_verdict_from_entry

    entry = {
        "title": "Es falso que el estudio sea verdadero según los expertos",
        "summary": "Análisis del caso.",
        "tags": [],
    }
    assert _infer_verdict_from_entry(entry, source_url="") == "UNVERIFIED"


def test_infer_verdict_unknown_domain_is_unverified():
    """Unknown domain without lexical signals → UNVERIFIED."""
    from services.etl.app.pipeline import _infer_verdict_from_entry

    entry = {
        "title": "Noticias del día en algún medio cualquiera",
        "summary": "Información variada sin contexto.",
        "tags": [],
    }
    assert _infer_verdict_from_entry(
        entry, source_url="https://some-random-blog.example.com/post"
    ) == "UNVERIFIED"


def test_infer_verdict_empty_url_falls_back_to_unverified():
    """Missing source_url must not break classification — defaults to UNVERIFIED."""
    from services.etl.app.pipeline import _infer_verdict_from_entry

    entry = {
        "title": "Noticia neutra sin pistas",
        "summary": "Sin palabras clave.",
        "tags": [],
    }
    assert _infer_verdict_from_entry(entry) == "UNVERIFIED"


def test_infer_verdict_www_prefix_is_normalized():
    """Domain matching must ignore 'www.' prefix."""
    from services.etl.app.pipeline import _infer_verdict_from_entry

    entry = {
        "title": "Press release from a national statistics institute",
        "summary": "New monthly CPI figures published.",
        "tags": [],
    }
    assert _infer_verdict_from_entry(
        entry, source_url="https://www.ine.es/some/path"
    ) == "REAL"


# ── extract ───────────────────────────────────────────────────────────────────

def test_extract_filters_invalid_entries():
    """extract() should return only articles that pass _validate_article."""
    from services.etl.app.pipeline import RSS_FEEDS, extract

    valid_entry_1 = {
        "title": "Artículo válido uno con título largo suficiente para pasar",
        "summary": "Este resumen tiene más de veinte caracteres y es completamente válido.",
        "link": "https://example.com/valid-1",
        "published": "2024-01-01",
        "tags": [],
    }
    valid_entry_2 = {
        "title": "Artículo válido dos también supera el umbral de longitud",
        "summary": "El segundo resumen también cumple el requisito de longitud mínima.",
        "link": "https://example.com/valid-2",
        "published": "2024-01-02",
        "tags": [],
    }
    invalid_entry = {
        "title": "Corto",       # 5 chars < 10
        "summary": "Muy breve",  # 9 chars < 20
        "link": "https://example.com/invalid",
        "published": "2024-01-03",
        "tags": [],
    }

    good_feed = MagicMock()
    good_feed.feed = {"title": "Test Feed"}
    good_feed.entries = [valid_entry_1, valid_entry_2, invalid_entry]

    empty_feed = MagicMock()
    empty_feed.entries = []

    def fake_parse(url):
        return good_feed if url == RSS_FEEDS[0] else empty_feed

    with patch("services.etl.app.pipeline.feedparser.parse", side_effect=fake_parse), \
         patch("services.etl.app.pipeline.extract_from_gnews", return_value=[]):
        result = extract()

    assert len(result) == 2
    urls = {a["url"] for a in result}
    assert "https://example.com/valid-1" in urls
    assert "https://example.com/valid-2" in urls


# ── extract_from_gnews ────────────────────────────────────────────────────────

def test_extract_from_gnews_deduplicates_urls():
    """extract_from_gnews() must deduplicate articles that share the same URL."""
    from services.etl.app.pipeline import extract_from_gnews

    article_a = {
        "url": "https://gnews.example/article-1",
        "title": "Bulo sobre vacunas desmentido por expertos sanitarios",
        "description": "Investigación de expertos sobre bulos de vacunas en redes.",
        "content": "Contenido completo del artículo sobre vacunas.",
        "source": {"name": "GNews Test"},
        "publishedAt": "2024-01-01T00:00:00Z",
    }
    article_b = {
        "url": "https://gnews.example/article-2",
        "title": "Desinformación climática analizada por científicos reconocidos",
        "description": "Análisis de expertos sobre desinformación climática global.",
        "content": "Contenido completo del artículo sobre clima.",
        "source": {"name": "GNews Test"},
        "publishedAt": "2024-01-02T00:00:00Z",
    }

    # 4 queries in extract_from_gnews: first returns 2 unique, second returns a duplicate
    responses = iter([
        {"articles": [article_a, article_b]},  # query 1: 2 unique articles
        {"articles": [article_a]},              # query 2: article_a duplicated
        {"articles": []},                       # query 3: empty
        {"articles": []},                       # query 4: empty
    ])

    mock_resp = MagicMock()
    mock_resp.json.side_effect = lambda: next(responses)

    with patch.dict(os.environ, {"GNEWS_API_KEY": "fake-test-key-12345"}), \
         patch("services.etl.app.pipeline.httpx.get", return_value=mock_resp):
        result = extract_from_gnews()

    assert len(result) == 2
    urls = {a["url"] for a in result}
    assert "https://gnews.example/article-1" in urls
    assert "https://gnews.example/article-2" in urls


def test_extract_from_gnews_returns_empty_without_api_key():
    """extract_from_gnews() returns [] immediately when GNEWS_API_KEY is not set."""
    from services.etl.app.pipeline import extract_from_gnews

    with patch("services.etl.app.pipeline.os.getenv", return_value=""):
        result = extract_from_gnews()

    assert result == []


# ── _ensure_collection_schema ─────────────────────────────────────────────────

def test_ensure_collection_schema_creates_when_absent():
    from services.etl.app.pipeline import _ensure_collection_schema

    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []  # no existing collections

    _ensure_collection_schema(mock_client, "fact_checks")

    mock_client.create_collection.assert_called_once()
    mock_client.delete_collection.assert_not_called()


def test_ensure_collection_schema_no_drop_when_present():
    from services.etl.app.pipeline import _ensure_collection_schema

    existing = MagicMock()
    existing.name = "fact_checks"
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = [existing]

    _ensure_collection_schema(mock_client, "fact_checks")

    mock_client.delete_collection.assert_not_called()
    mock_client.create_collection.assert_not_called()


def test_ensure_collection_schema_force_recreate():
    from services.etl.app.pipeline import _ensure_collection_schema

    existing = MagicMock()
    existing.name = "fact_checks"
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = [existing]

    _ensure_collection_schema(mock_client, "fact_checks", force_recreate=True)

    mock_client.delete_collection.assert_called_once_with("fact_checks")
    mock_client.create_collection.assert_called_once()
