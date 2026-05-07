"""
ETL pipeline — keeps the Qdrant knowledge base up to date with recent
fact-checking articles.

Stages:
  Extract   → Pull articles from fact-checking RSS feeds / Google API.
  Transform → Clean text, generate dense (BAAI/bge-m3) + sparse (BM25) vectors.
  Load      → Upsert Points into Qdrant collection.

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
from qdrant_client.models import PointStruct, VectorParams, Distance

load_dotenv()

# Fact-checking RSS sources (extend freely)
RSS_FEEDS = [
    "https://maldita.es/feed/",
    "https://newtral.es/feed/",
    "https://www.snopes.com/feed/",
]


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
            })
    return articles


def transform(articles: list[dict]):
    """Generate dense embeddings for each article. Returns list of PointStruct."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))

    points = []
    for art in articles:
        text = f"{art['title']} {art['summary']}"
        vector = model.encode(text).tolist()
        points.append(
            PointStruct(
                id=art["id"],
                vector=vector,
                payload={k: v for k, v in art.items() if k != "id"},
            )
        )
    return points


def load(points) -> None:
    """Upsert points into Qdrant."""
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )
    collection = os.getenv("QDRANT_COLLECTION", "fact_checks")

    # Create collection if it doesn't exist
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

    client.upsert(collection_name=collection, points=points)
    print(f"[etl] Upserted {len(points)} points into '{collection}'.")


def run() -> None:
    print("[etl] Starting pipeline…")
    articles = extract()
    print(f"[etl] Extracted {len(articles)} articles.")
    points = transform(articles)
    load(points)
    print("[etl] Done.")


if __name__ == "__main__":
    run()
