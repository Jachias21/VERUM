"""
ETL pipeline — keeps the Qdrant knowledge base up to date with recent
fact-checking articles.

Stages:
  Extract   → Pull articles from fact-checking RSS feeds / Google API.
  Transform → Clean text, generate dense (intfloat/multilingual-e5-large) + sparse (Qdrant/bm25) vectors.
  Load      → Upsert Points into Qdrant collection (named-vector / hybrid schema).

Run modes:
  - One-shot: python -m app.pipeline
  - Scheduled: deploy as a Docker service with a cron or Airflow DAG.
"""
from __future__ import annotations

import logging
import os
import socket
import ssl
import uuid

import feedparser
import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

load_dotenv()

logger = logging.getLogger("verum.etl")

from .seed_classics import CLASSIC_HOAXES  # noqa: E402

# Fact-checking RSS sources (extend freely)
RSS_FEEDS = [
    # ── Spanish fact-checkers ─────────────────────────────────────────
    "https://maldita.es/feed/",
    "https://newtral.es/feed/",
    "https://verificat.cat/feed/",
    "https://factual.afp.com/list/es/rss.xml",  # possibly down — keep for retry
    "https://www.newtral.es/area/fast-forward/feed/",
    "https://hechosdehoy.com/feed/",
    # ── Latin-American fact-checkers ────────────────────────────────
    "https://chequeado.com/feed/",          # possibly down — keep for retry
    "https://colombiacheck.com/feed",       # possibly down — keep for retry
    "https://factchequeado.com/feed/",
    "https://lasillavacia.com/feed",
    "https://www.pagina12.com.ar/rss/secciones/el-planeta/notas",  # possibly down — keep for retry
    # ── English fact-checkers (high quality) ────────────────────────
    "https://www.snopes.com/feed/",
    "https://www.factcheck.org/feed/",
    "https://fullfact.org/feed/",
    "https://apnews.com/hub/fact-checking/feed",  # possibly down — keep for retry
    "https://www.politifact.com/rss/all.rss/",    # possibly down — keep for retry
    "https://factuel.afp.com/list/en/rss.xml",    # possibly down — keep for retry
    # ── Anti-disinformation organisations ─────────────────────────
    "https://www.stopfake.org/en/feed/",
    "https://euvsdisinfo.eu/feed/",
    # ── Reference media with fact-check sections ───────────────────
    "https://elpais.com/rss/elpais/portada.xml",
    "https://www.bbc.com/mundo/rss.xml",
    "https://www.20minutos.es/rss/",
    "https://www.efeverde.com/feed/",
    # ── General quality news (broad coverage) ─────────────────────
    "https://www.lavanguardia.com/mvc/feed/rss/vida/salud",  # possibly down — keep for retry
    "https://www.elmundo.es/rss/espana.xml",
    "https://www.rtve.es/api/noticias.rss",         # possibly down — keep for retry
    "https://www.infobae.com/feeds/rss/",           # possibly down — keep for retry
    "https://www.clarin.com/rss/lo-ultimo/",
    # ── Official institutional sources (Spain) ─────────────────────
    "https://www.lamoncloa.gob.es/serviciosdeprensa/notasprensa/Paginas/index.aspx?format=rss",  # possibly down — keep for retry
    "https://www.boe.es/rss/canal.php?c=ultimas_disposiciones",          # possibly down — keep for retry
    "https://www.sanidad.gob.es/rss/notasPrensa.do",                     # possibly down — keep for retry
    "https://www.educacionyfp.gob.es/prensa/actualidad.rss",             # possibly down — keep for retry
    # ── Official institutional sources (international) ─────────────
    "https://www.un.org/feed/subscribe/en/rss/category/un-news/feed",  # possibly down — keep for retry
    "https://www.who.int/rss-feeds/news-english.xml",
    "https://www.esa.int/rssfeed/Our_Activities",
    "https://ec.europa.eu/commission/presscorner/api/rss?language=es",
    # ── General news agencies (broad factual coverage) ────────────
    "https://www.efe.com/efe/espana/1/rss",         # possibly down — keep for retry
    "https://feeds.bbci.co.uk/news/world/rss.xml",  # possibly down — keep for retry
    "https://apnews.com/index.rss",                 # possibly down — keep for retry
    "https://www.dw.com/es/top-stories/s-30684/rss",  # possibly down — keep for retry
    # ── Science & health (peer-reviewed coverage) ─────────────────
    "https://www.nature.com/nature.rss",
    "https://www.nih.gov/news-events/news-releases/feed",  # possibly down — keep for retry
    "https://www.cdc.gov/media/rss/index.htm",             # possibly down — keep for retry
    # ── Economy (institutional and quality general) ───────────────
    "https://www.bde.es/rss/notas_es.xml",         # possibly down — keep for retry
    "https://cincodias.elpais.com/rss/cincodias/portada.xml",
    "https://www.expansion.com/rss/portada.xml",
]

# Domains of publishers that publish factual information from institutional/official sources.
# Match is performed against the netloc of the feed URL (or article URL for GNews),
# which is deterministic — unlike feed titles, which vary between publishers.
# Articles from these domains without explicit FAKE keywords default to REAL.
_INSTITUTIONAL_DOMAINS = {
    # Spain — institutional
    "lamoncloa.gob.es",
    "boe.es",
    "ine.es",
    "sanidad.gob.es",
    "educacionyfp.gob.es",
    "exteriores.gob.es",
    "bde.es",
    # International — institutional
    "un.org",
    "who.int",
    "esa.int",
    "ecb.europa.eu",
    "ec.europa.eu",
    "nih.gov",
    "cdc.gov",
    # News agencies (broad factual coverage)
    "efe.com",
    "efeverde.com",      # EFE Verde — environmental section of Spain's national agency
    "reuters.com",
    "bbc.com",
    "bbc.co.uk",
    "bbci.co.uk",
    "apnews.com",
    "dw.com",
    # Science publishers (peer-reviewed coverage)
    "nature.com",
    "science.org",
    "sciencemag.org",
    # Quality economy outlets
    "cincodias.elpais.com",
    "expansion.com",
    # Spain — reference quality dailies
    "elpais.com",        # El País — Spain's newspaper of record
    "elmundo.es",        # El Mundo — major Spanish daily
    "abc.es",            # ABC — major Spanish daily
    "lavanguardia.com",  # La Vanguardia — major Catalan/Spanish daily
    "20minutos.es",      # 20 Minutos — widely distributed free daily
    "rtve.es",           # RTVE — Spain's public broadcaster
    # Latin America — reference quality dailies
    "clarin.com",        # Clarín — Argentina's most-read newspaper
    "infobae.com",       # Infobae — major Latin-American digital outlet
    "pagina12.com.ar",   # Página 12 — Argentine daily of record
    "lanacion.com.ar",   # La Nación — Argentina's newspaper of record
}

# Domains of fact-checking publishers. Match against feed URL netloc.
# Articles from these without explicit FAKE/REAL keywords default to UNVERIFIED
# (their content may be context/analysis without a verdict).
_FACTCHECKER_DOMAINS = {
    "maldita.es",
    "newtral.es",
    "verificat.cat",
    "factual.afp.com",
    "factuel.afp.com",
    "hechosdehoy.com",
    "chequeado.com",
    "colombiacheck.com",
    "factchequeado.com",
    "lasillavacia.com",
    "snopes.com",
    "factcheck.org",
    "fullfact.org",
    "politifact.com",
    "stopfake.org",
    "euvsdisinfo.eu",
    "aosfatos.org",
    "ecuadorchequea.com",
}

# Dense vector dimension for BAAI/bge-m3
DENSE_DIM = 1024


_FAKE_KEYWORDS = [
    "falso", "false", "fake", "bulo", "hoax", "incorrecto", "desinformación",
    "engaño", "manipulado", "mentira", "no es cierto", "sin evidencia",
    "misleading", "misinformation", "debunked", "desmentido",
]
_REAL_KEYWORDS = [
    "verdadero", "true", "correcto", "verified", "confirmado", "cierto", "demostrado",
]


from urllib.parse import urlparse


def _domain_class(source_url: str) -> str:
    """Classify a URL by publisher domain. Returns 'institutional', 'factchecker', or 'unknown'."""
    if not source_url:
        return "unknown"
    try:
        netloc = urlparse(source_url).netloc.lower()
    except Exception:
        return "unknown"
    if not netloc:
        return "unknown"
    # Strip leading 'www.' for matching
    if netloc.startswith("www."):
        netloc = netloc[4:]
    # Exact match or any subdomain of a known domain (e.g. 'feeds.reuters.com' matches 'reuters.com')
    for inst in _INSTITUTIONAL_DOMAINS:
        if netloc == inst or netloc.endswith("." + inst):
            return "institutional"
    for fc in _FACTCHECKER_DOMAINS:
        if netloc == fc or netloc.endswith("." + fc):
            return "factchecker"
    return "unknown"


def _infer_verdict_from_entry(
    entry: dict,
    publisher: str = "",
    source_url: str = "",
) -> str:
    """Best-effort verdict using title, tags, summary text, and publisher domain.

    Decision flow:
      1. Lexical signals dominate: explicit FAKE/REAL keywords in text always win.
      2. If both kinds of signals appear → UNVERIFIED (ambiguous).
      3. Without explicit signals, fall back to publisher domain class:
         - institutional → REAL (editorial mandate is verified facts).
         - factchecker → UNVERIFIED (content may be context, not always a verdict).
         - unknown → UNVERIFIED (safe default).

    The `publisher` parameter is kept for backward compatibility but is no
    longer used for classification — `source_url` (passed by the caller from
    the feed/article URL) is the authoritative signal.
    """
    tags = " ".join(t.get("term", "") for t in entry.get("tags", [])).lower()
    title = entry.get("title", "").lower()
    raw_summary = entry.get("summary", "")
    clean_summary = BeautifulSoup(raw_summary, "html.parser").get_text(separator=" ", strip=True).lower()
    combined = " ".join([tags, title, clean_summary])

    has_fake = any(w in combined for w in _FAKE_KEYWORDS)
    has_real = any(w in combined for w in _REAL_KEYWORDS)

    if has_fake and has_real:
        return "UNVERIFIED"
    if has_fake:
        return "FAKE"
    if has_real:
        return "REAL"

    domain_class = _domain_class(source_url)
    if domain_class == "institutional":
        return "REAL"
    if domain_class == "factchecker":
        return "UNVERIFIED"
    return "UNVERIFIED"


def extract_from_gnews() -> list[dict]:
    """Pull articles from GNews API for multiple queries (Spanish + English).

    Returns an empty list silently if GNEWS_API_KEY is not set or the request
    fails, so the rest of the ETL is never disrupted.
    Deduplicates by URL across queries.
    """
    api_key = os.getenv("GNEWS_API_KEY", "")
    if not api_key:
        return []

    queries = ["fake news bulo", "desinformación España", "fact check", "hoax viral"]
    seen: dict[str, dict] = {}  # url → article dict (dedup)

    for query in queries:
        url = (
            f"https://gnews.io/api/v4/search"
            f"?q={query.replace(' ', '+')}&lang=es&token={api_key}&max=20"
        )
        try:
            resp = httpx.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"[etl] GNews request failed for query {query!r}, skipping: {exc}")
            continue

        for item in data.get("articles", []):
            link = item.get("url", "")
            if not link or link in seen:
                continue
            publisher = item.get("source", {}).get("name", "GNews")
            raw_summary = item.get("description", "") or ""
            clean_summary = BeautifulSoup(raw_summary, "html.parser").get_text(separator=" ", strip=True)
            synthetic_entry = {
                "title": item.get("title", ""),
                "summary": item.get("content", raw_summary),
                "tags": [],
            }
            seen[link] = {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, link)),
                "title": item.get("title", ""),
                "summary": clean_summary,
                "url": link,
                "publisher": publisher,
                "published": item.get("publishedAt", ""),
                "verdict": _infer_verdict_from_entry(synthetic_entry, publisher=publisher, source_url=link),
            }

    return list(seen.values())


def _validate_article(article: dict) -> bool:
    """Return True only if the article has a usable title, summary, and URL."""
    title = article.get("title", "")
    summary = article.get("summary", "")
    url = article.get("url", "")
    return len(title) >= 10 and len(summary) >= 20 and url.startswith("http")


def extract() -> list[dict]:
    """Pull latest articles from RSS feeds and GNews API."""
    articles: list[dict] = []
    feeds_ok = 0
    feeds_empty = 0
    feeds_failed = 0
    raw_count = 0

    for url in RSS_FEEDS:
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(15)
        try:
            feed = feedparser.parse(url)
            if not feed.entries:
                feeds_empty += 1
                continue
            feeds_ok += 1
            publisher = feed.feed.get("title", "")
            for entry in feed.entries:
                raw_summary = entry.get("summary", "")
                clean_summary = BeautifulSoup(raw_summary, "html.parser").get_text(separator=" ", strip=True)
                article = {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, entry.get("link", ""))),
                    "title": entry.get("title", ""),
                    "summary": clean_summary,
                    "url": entry.get("link", ""),
                    "publisher": publisher,
                    "published": entry.get("published", ""),
                    "verdict": _infer_verdict_from_entry(entry, publisher=publisher, source_url=url),
                }
                raw_count += 1
                if _validate_article(article):
                    articles.append(article)
        except (socket.timeout, TimeoutError):
            print(f"[etl] TIMEOUT: {url}")
            feeds_failed += 1
        except ssl.SSLError:
            print(f"[etl] SSL_ERROR: {url}")
            feeds_failed += 1
        except Exception as exc:
            print(f"[etl] FAILED ({type(exc).__name__}): {url} — {exc}")
            feeds_failed += 1
        finally:
            socket.setdefaulttimeout(old_timeout)

    discarded = raw_count - len(articles)
    if discarded:
        print(f"[etl] Discarded {discarded} articles with empty title/summary/url")
    print(f"[etl] Feed summary: {feeds_ok} OK / {feeds_empty} empty / {feeds_failed} failed")

    gnews_articles = extract_from_gnews()
    if gnews_articles:
        print(f"[etl] GNews contributed {len(gnews_articles)} articles (across all queries).")
    articles.extend(gnews_articles)
    return articles


BATCH_SIZE = 10  # max articles per embedding + upsert cycle to avoid OOM


def _ensure_collection_schema(client: QdrantClient, collection: str, force_recreate: bool = False) -> None:
    """Ensure the Qdrant collection exists with the correct hybrid schema.

    By default uses incremental upsert mode: if the collection already exists
    it is left untouched so the bot keeps serving requests during ETL runs.
    Pass force_recreate=True only for schema migrations.
    """
    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        if force_recreate:
            print(f"[etl] force_recreate=True — dropping existing collection '{collection}'…")
            client.delete_collection(collection)
        else:
            print(f"[etl] Collection already exists, using upsert mode")
            return
    print(f"[etl] Creating hybrid collection '{collection}' (dense + sparse)…")
    client.create_collection(
        collection_name=collection,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            ),
        },
    )


def _embed_batch(
    articles: list[dict],
    dense_model,
    sparse_model,
) -> list[PointStruct]:
    """Embed a single batch and return PointStructs. Does NOT load models."""
    import gc

    texts = [f"{art['title']} {art['summary']}" for art in articles]

    dense_vectors = list(dense_model.embed(texts))
    sparse_embeddings = list(sparse_model.embed(texts))

    points = []
    for art, dense_vec, sparse_emb in zip(articles, dense_vectors, sparse_embeddings):
        points.append(
            PointStruct(
                id=art["id"],
                vector={
                    "dense": dense_vec.tolist(),
                    "sparse": SparseVector(
                        indices=sparse_emb.indices.tolist(),
                        values=sparse_emb.values.tolist(),
                    ),
                },
                payload={k: v for k, v in art.items() if k != "id"},
            )
        )

    # Free intermediate tensors immediately
    del texts, dense_vectors, sparse_embeddings
    gc.collect()
    return points


# Keep transform() and load() for backwards-compat / direct calls
def transform(articles: list[dict]) -> list[PointStruct]:
    """Generate dense + sparse embeddings for each article. Returns list of PointStruct."""
    from fastembed import TextEmbedding, SparseTextEmbedding
    import gc

    print("[etl] Loading dense model (multilingual-e5-large, ONNX)…")
    dense_model = TextEmbedding(model_name="intfloat/multilingual-e5-large", threads=2)
    print("[etl] Loading sparse model (Qdrant/bm25)…")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25", threads=2)

    points = _embed_batch(articles, dense_model, sparse_model)

    del dense_model, sparse_model
    gc.collect()
    return points


def load(points: list[PointStruct]) -> None:
    """Ensure collection schema and upsert all points (incremental by default)."""
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )
    collection = os.getenv("QDRANT_COLLECTION", "fact_checks")
    _ensure_collection_schema(client, collection)
    client.upsert(collection_name=collection, points=points)
    print(f"[etl] Upserted {len(points)} points into '{collection}' (incremental mode).")


def run() -> None:
    import gc
    from fastembed import TextEmbedding, SparseTextEmbedding

    print("[etl] Starting hybrid ETL pipeline…")
    articles = extract()
    print(f"[etl] Extracted {len(articles)} articles (valid).")

    # ── Setup Qdrant (collection created only if absent; upsert mode otherwise) ─
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )
    collection = os.getenv("QDRANT_COLLECTION", "fact_checks")
    _ensure_collection_schema(client, collection)

    # ── Seed classic hoaxes if requested or collection is nearly empty ────────
    _seed_flag = os.getenv("SEED_CLASSICS", "true").strip().lower()
    _should_seed = _seed_flag in ("1", "true", "yes")
    if not _should_seed:
        try:
            _point_count = client.count(collection_name=collection).count
            _should_seed = _point_count < 5
        except Exception:
            _should_seed = False
    if _should_seed:
        logger.info("[etl] Seeding %d classic hoaxes into Qdrant", len(CLASSIC_HOAXES))
        print(f"[etl] Seeding {len(CLASSIC_HOAXES)} classic hoaxes into Qdrant…")
        seed_articles = [
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, h["url"])),
                "title": h["title"],
                "summary": h["content_summary"],
                "url": h["url"],
                "publisher": h["source_publisher"],
                "published": h["publish_date"],
                "verdict": h["verdict"],
            }
            for h in CLASSIC_HOAXES
        ]
        articles = seed_articles + articles

    # ── Load models once — kept alive across batches ────────────────────────
    print("[etl] Loading dense model (multilingual-e5-large, ONNX)…")
    dense_model = TextEmbedding(model_name="intfloat/multilingual-e5-large")

    # Free dense model RAM before loading sparse (peak-RAM reduction)
    # NOTE: we keep it alive only while encoding; sparse is loaded after.
    print("[etl] Loading sparse model (Qdrant/bm25)…")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    total = len(articles)
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    total_upserted = 0

    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        batch = articles[start: start + BATCH_SIZE]
        print(f"[etl] Procesando y guardando lote {batch_idx + 1}/{num_batches} "
              f"({len(batch)} artículos)…")

        points = _embed_batch(batch, dense_model, sparse_model)
        client.upsert(collection_name=collection, points=points)
        total_upserted += len(points)

        del points
        gc.collect()

    # Release models
    del dense_model, sparse_model
    gc.collect()

    print(f"[etl] Upserted {total_upserted} points into '{collection}' (incremental mode).")

    # ── Publisher summary (top 10) ────────────────────────────────────────
    from collections import Counter
    counts = Counter(art["publisher"] for art in articles)
    print("[etl] Articles per publisher (top 10):")
    for publisher, count in counts.most_common(10):
        print(f"  {count:>5}  {publisher}")

    print("[etl] Done.")


if __name__ == "__main__":
    run()
