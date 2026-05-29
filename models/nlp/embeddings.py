"""
Utilidades de modelo de embeddings compartidas entre el pipeline ETL y worker_nlp.

Modelo denso  : intfloat/multilingual-e5-large (multilingual, 1024-dim, ONNX vía fastembed).
Modelo sparse : Qdrant/bm25 vía fastembed (vectores sparse BM25).
"""
from __future__ import annotations

_dense_model = None   # fastembed.TextEmbedding - lazy
_sparse_model = None  # fastembed.SparseTextEmbedding - lazy

DENSE_MODEL_NAME = "intfloat/multilingual-e5-large"


def get_model():
    global _dense_model
    if _dense_model is None:
        from fastembed import TextEmbedding
        _dense_model = TextEmbedding(model_name=DENSE_MODEL_NAME)
    return _dense_model


def embed(texts: list[str]) -> list[list[float]]:
    """Devuelve embeddings densos. Añade el prefijo 'query: ' para elementos individuales (consultas RAG)."""
    model = get_model()
    return [v.tolist() for v in model.embed(texts)]


def get_sparse_model():
    global _sparse_model
    if _sparse_model is None:
        from fastembed import SparseTextEmbedding
        _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _sparse_model


def sparse_embed(texts: list[str]):
    """Devuelve una lista de objetos SparseEmbedding de fastembed (con .indices y .values)."""
    return list(get_sparse_model().embed(texts))
