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
]

# Dense vector dimension for BAAI/bge-m3
DENSE_DIM = 1024


def _infer_verdict_from_entry(entry: dict) -> str:
    """Best-effort verdict from RSS tags/categories before storing in Qdrant."""
    tags = " ".join(t.get("term", "") for t in entry.get("tags", [])).lower()
    title = entry.get("title", "").lower()
    combined = tags + " " + title
    if any(w in combined for w in ["falso", "false", "fake", "bulo", "hoax", "incorrecto"]):
        return "FAKE"
    if any(w in combined for w in ["verdadero", "true", "correcto", "verified"]):
        return "REAL"
    return "UNVERIFIED"


def extract() -> list[dict]:
    """Pull latest articles from RSS feeds."""
    articles = []
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            raw_summary = entry.get("summary", "")
            clean_summary = BeautifulSoup(raw_summary, "html.parser").get_text(separator=" ", strip=True)
            articles.append({
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, entry.get("link", ""))),
                "title": entry.get("title", ""),
                "summary": clean_summary,
                "url": entry.get("link", ""),
                "publisher": feed.feed.get("title", ""),
                "published": entry.get("published", ""),
                "verdict": _infer_verdict_from_entry(entry),
            })
    return articles


def transform(articles: list[dict]) -> list[PointStruct]:
    """Generate dense + sparse embeddings for each article. Returns list of PointStruct."""
    from fastembed import TextEmbedding, SparseTextEmbedding

    texts = [f"{art['title']} {art['summary']}" for art in articles]

    print("[etl] Loading dense model (multilingual-e5-large, ONNX)…")
    dense_model = TextEmbedding(model_name="intfloat/multilingual-e5-large")
    print(f"[etl] Encoding {len(texts)} texts (dense)…")
    dense_vectors = list(dense_model.embed(texts))

    # Free dense model before loading sparse to reduce peak RAM usage
    import gc
    del dense_model
    gc.collect()

    print("[etl] Loading sparse model (Qdrant/bm25)…")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    print(f"[etl] Encoding {len(texts)} texts (sparse)…")
    sparse_embeddings = list(sparse_model.embed(texts))

    points = []
    for art, dense_vec, sparse_emb in zip(articles, dense_vectors, sparse_embeddings):
        sparse_vec = SparseVector(
            indices=sparse_emb.indices.tolist(),
            values=sparse_emb.values.tolist(),
        )
        points.append(
            PointStruct(
                id=art["id"],
                vector={
                    "dense": dense_vec.tolist(),
                    "sparse": sparse_vec,
                },
                payload={k: v for k, v in art.items() if k != "id"},
            )
        )
    return points


def load(points: list[PointStruct]) -> None:
    """Recreate collection with hybrid schema and upsert all points."""
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )
    collection = os.getenv("QDRANT_COLLECTION", "fact_checks")

    # Always recreate so the schema reflects dense + sparse named vectors
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

    client.upsert(collection_name=collection, points=points)
    print(f"[etl] Upserted {len(points)} hybrid points into '{collection}'.")


def run() -> None:
    print("[etl] Starting hybrid ETL pipeline…")
    articles = extract()
    print(f"[etl] Extracted {len(articles)} articles.")
    points = transform(articles)
    load(points)
    print("[etl] Done.")


if __name__ == "__main__":
    run()
