"""
Embedding model utilities shared between the ETL pipeline and worker_nlp.

Model: BAAI/bge-m3 (multilingual, supports dense + sparse vectors natively).
"""
from __future__ import annotations

import os

from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    return get_model().encode(texts, normalize_embeddings=True).tolist()
