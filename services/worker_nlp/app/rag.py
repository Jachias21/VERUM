"""
RAG pipeline — Hybrid Search (Local Qdrant → Google Fact Check API fallback)
followed by LLM synthesis via Ollama.

Architecture:
  Level 1: BM25 + dense vector search against local Qdrant knowledge base.
  Level 2: On-demand Google Fact Check Tools API (if Level 1 confidence < threshold).
  Fusion:  Reciprocal Rank Fusion (RRF) of both result sets.
  Synthesis: Ollama SLM (Llama 3.2 / Qwen 2.5) generates a 3-line verdict.
"""
from __future__ import annotations

import os
import uuid

import httpx
from shared.schemas import NLPResult


async def hybrid_search(
    query_id: uuid.UUID,
    text: str,
    entities: list[str],
) -> NLPResult:
    """
    Execute Level-1 local search; fall back to Level-2 API if needed.
    Returns a partially-filled NLPResult (summary is filled by synthesize_verdict).
    """
    threshold = float(os.getenv("NLP_CONFIDENCE_THRESHOLD", 0.75))

    # ── Level 1: Local Qdrant hybrid search ──────────────────────────────────
    local_hits = await _search_qdrant(entities)

    if local_hits and local_hits[0]["score"] >= threshold:
        best = local_hits[0]
        return NLPResult(
            query_id=query_id,
            extracted_entities=entities,
            fact_check_matches=len(local_hits),
            source_url=best.get("url"),
            verdict=best.get("verdict", "UNVERIFIED"),
            summary="",   # filled later by synthesize_verdict
        )

    # ── Level 2: Google Fact Check Tools API fallback ─────────────────────────
    api_hits = await _search_google_fact_check(" ".join(entities))

    if not api_hits:
        return NLPResult(
            query_id=query_id,
            extracted_entities=entities,
            fact_check_matches=0,
            verdict="UNVERIFIED",
            summary="",
        )

    best = api_hits[0]
    return NLPResult(
        query_id=query_id,
        extracted_entities=entities,
        fact_check_matches=len(api_hits),
        source_url=best.get("url"),
        verdict=best.get("verdict", "UNVERIFIED"),
        summary="",
    )


async def synthesize_verdict(text: str, result: NLPResult) -> str:
    """
    Send raw results to Ollama and ask the SLM to write a 3-line user verdict.
    Temperature is kept very low to avoid hallucination.

    TODO: replace placeholder with actual LangChain/Ollama chain call.
    """
    # TODO: build prompt, call OllamaLLM via LangChain, return response
    return (
        f"Veredicto: {result.verdict}\n"
        f"Fuentes consultadas: {result.fact_check_matches}\n"
        f"Referencia: {result.source_url or 'Sin coincidencias en la base de datos.'}"
    )


# ── Private helpers ───────────────────────────────────────────────────────────

async def _search_qdrant(entities: list[str]) -> list[dict]:
    """BM25 + dense vector hybrid search against local Qdrant collection."""
    # TODO: embed query with BAAI/bge-m3, perform hybrid search via qdrant-client
    return []


async def _search_google_fact_check(query: str) -> list[dict]:
    """Call Google Fact Check Tools API and normalise results."""
    api_key = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "")
    if not api_key:
        return []
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params={"query": query, "key": api_key, "languageCode": "es"})
    if resp.status_code != 200:
        return []
    claims = resp.json().get("claims", [])
    results = []
    for claim in claims:
        review = claim.get("claimReview", [{}])[0]
        results.append({
            "score": 1.0,   # API results always treated as high confidence
            "verdict": _normalise_rating(review.get("textualRating", "")),
            "url": review.get("url", ""),
        })
    return results


def _normalise_rating(rating: str) -> str:
    rating_lower = rating.lower()
    if any(w in rating_lower for w in ["falso", "false", "fake", "incorrect", "incorrecto"]):
        return "FAKE"
    if any(w in rating_lower for w in ["verdadero", "true", "correct", "correcto"]):
        return "REAL"
    return "UNVERIFIED"
