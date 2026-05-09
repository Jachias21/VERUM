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

import os
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

# Fact-checking RSS sources (extend freely)
RSS_FEEDS = [
    "https://maldita.es/feed/",
    "https://newtral.es/feed/",
    "https://www.snopes.com/feed/",
    "https://verificat.cat/feed/",
    "https://factual.afp.com/list/es/rss.xml",
    "https://www.reuters.com/fact-check/rss.xml",
    "https://www.efeverde.com/feed/",
    "https://www.newtral.es/area/fast-forward/feed/",
    "https://hechosdehoy.com/feed/",
]

# Publishers that exclusively publish fact-checks / debunks → default verdict FAKE
_FACTCHECKER_PUBLISHERS = [
    "maldita", "newtral", "verificat", "afp factual", "fact check", "reuters",
]

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


def _infer_verdict_from_entry(entry: dict, publisher: str = "") -> str:
    """Best-effort verdict using title, tags, and summary text before storing in Qdrant."""
    tags = " ".join(t.get("term", "") for t in entry.get("tags", [])).lower()
    title = entry.get("title", "").lower()
    raw_summary = entry.get("summary", "")
    clean_summary = BeautifulSoup(raw_summary, "html.parser").get_text(separator=" ", strip=True).lower()
    combined = " ".join([tags, title, clean_summary])

    if any(w in combined for w in _FAKE_KEYWORDS):
        return "FAKE"
    if any(w in combined for w in _REAL_KEYWORDS):
        return "REAL"

    # Heuristic: known fact-checker publishers only publish debunks → default FAKE
    pub_lower = publisher.lower()
    if any(fc in pub_lower for fc in _FACTCHECKER_PUBLISHERS):
        return "FAKE"

    return "UNVERIFIED"


def extract_from_gnews() -> list[dict]:
    """Pull articles from GNews API (Spanish fake-news search).

    Returns an empty list silently if GNEWS_API_KEY is not set or the request
    fails, so the rest of the ETL is never disrupted.
    """
    api_key = os.getenv("GNEWS_API_KEY", "")
    if not api_key:
        return []

    url = (
        f"https://gnews.io/api/v4/search"
        f"?q=fake+news+bulo&lang=es&token={api_key}&max=20"
    )
    try:
        resp = httpx.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"[etl] GNews request failed, skipping: {exc}")
        return []

    articles = []
    for item in data.get("articles", []):
        link = item.get("url", "")
        publisher = item.get("source", {}).get("name", "GNews")
        raw_summary = item.get("description", "") or ""
        clean_summary = BeautifulSoup(raw_summary, "html.parser").get_text(separator=" ", strip=True)
        # Build a synthetic entry dict compatible with _infer_verdict_from_entry
        synthetic_entry = {
            "title": item.get("title", ""),
            "summary": item.get("content", raw_summary),
            "tags": [],
        }
        articles.append({
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, link)),
            "title": item.get("title", ""),
            "summary": clean_summary,
            "url": link,
            "publisher": publisher,
            "published": item.get("publishedAt", ""),
            "verdict": _infer_verdict_from_entry(synthetic_entry, publisher),
        })
    return articles


def extract() -> list[dict]:
    """Pull latest articles from RSS feeds and GNews API."""
    articles = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            publisher = feed.feed.get("title", "")
            for entry in feed.entries:
                raw_summary = entry.get("summary", "")
                clean_summary = BeautifulSoup(raw_summary, "html.parser").get_text(separator=" ", strip=True)
                articles.append({
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, entry.get("link", ""))),
                    "title": entry.get("title", ""),
                    "summary": clean_summary,
                    "url": entry.get("link", ""),
                    "publisher": publisher,
                    "published": entry.get("published", ""),
                    "verdict": _infer_verdict_from_entry(entry, publisher),
                })
        except Exception as exc:
            print(f"[etl] Failed to parse feed {url!r}, skipping: {exc}")

    gnews_articles = extract_from_gnews()
    if gnews_articles:
        print(f"[etl] GNews contributed {len(gnews_articles)} articles.")
    articles.extend(gnews_articles)
    return articles


BATCH_SIZE = 20  # max articles per embedding + upsert cycle to avoid OOM


def _ensure_collection(client: QdrantClient, collection: str) -> None:
    """Drop-and-recreate the Qdrant collection so the schema is always current."""
    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        print(f"[etl] Dropping existing collection '{collection}'…")
        client.delete_collection(collection)
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
    dense_model = TextEmbedding(model_name="intfloat/multilingual-e5-large")
    print("[etl] Loading sparse model (Qdrant/bm25)…")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    points = _embed_batch(articles, dense_model, sparse_model)

    del dense_model, sparse_model
    gc.collect()
    return points


def load(points: list[PointStruct]) -> None:
    """Recreate collection with hybrid schema and upsert all points."""
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )
    collection = os.getenv("QDRANT_COLLECTION", "fact_checks")
    _ensure_collection(client, collection)
    client.upsert(collection_name=collection, points=points)
    print(f"[etl] Upserted {len(points)} hybrid points into '{collection}'.")


def run() -> None:
    import gc
    from fastembed import TextEmbedding, SparseTextEmbedding

    print("[etl] Starting hybrid ETL pipeline…")
    articles = extract()
    print(f"[etl] Extracted {len(articles)} articles.")

    # ── Setup Qdrant (collection recreated once, before any upserts) ────────
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )
    collection = os.getenv("QDRANT_COLLECTION", "fact_checks")
    _ensure_collection(client, collection)

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

    print(f"[etl] Upserted {total_upserted} hybrid points into '{collection}'.")
    print("[etl] Done.")


if __name__ == "__main__":
    run()
