"""
Embedding model utilities shared between the ETL pipeline and worker_nlp.

Dense model : intfloat/multilingual-e5-large (multilingual, 1024-dim, ONNX via fastembed).
Sparse model: Qdrant/bm25 via fastembed (BM25 sparse vectors).
"""
from __future__ import annotations

_dense_model = None   # fastembed.TextEmbedding — lazy
_sparse_model = None  # fastembed.SparseTextEmbedding — lazy

DENSE_MODEL_NAME = "intfloat/multilingual-e5-large"


def get_model():
    global _dense_model
    if _dense_model is None:
        from fastembed import TextEmbedding
        _dense_model = TextEmbedding(model_name=DENSE_MODEL_NAME)
    return _dense_model


def embed(texts: list[str]) -> list[list[float]]:
    """Return dense embeddings. Prefixes 'query: ' for single items (RAG queries)."""
    model = get_model()
    return [v.tolist() for v in model.embed(texts)]


def get_sparse_model():
    global _sparse_model
    if _sparse_model is None:
        from fastembed import SparseTextEmbedding
        _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _sparse_model


def sparse_embed(texts: list[str]):
    """Return a list of fastembed SparseEmbedding objects (have .indices, .values)."""
    return list(get_sparse_model().embed(texts))
