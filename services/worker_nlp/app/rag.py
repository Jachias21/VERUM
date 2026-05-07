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

import asyncio
from http import client
import os
import uuid

import httpx
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from qdrant_client import AsyncQdrantClient
from shared.schemas import NLPResult
from models.nlp.embeddings import embed


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
            summary=best.get("text", ""),  # article text; overwritten by synthesize_verdict
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
    Send the viral message and the retrieved fact-checking article to Ollama
    and ask the SLM to produce a 3-4 line verdict.
    Temperature is kept at 0 to minimise hallucination.
    """
    retrieved_article = result.summary  # populated by hybrid_search

    if not retrieved_article:
        return (
            f"Veredicto: {result.verdict}\n"
            f"Fuentes consultadas: {result.fact_check_matches}\n"
            f"Referencia: {result.source_url or 'Sin coincidencias en la base de datos.'}"
        )

    ollama_url = os.getenv("OLLAMA_HOST", "http://ollama:11434")
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    prompt = PromptTemplate(
        input_variables=["message_text", "retrieved_article"],
        template=(
            "Eres un verificador de hechos objetivo e imparcial.\n"
            "Tu ÚNICA fuente de información es el artículo de referencia que se te proporciona.\n"
            "NO inventes datos ni uses conocimiento externo al artículo.\n\n"
            "MENSAJE VIRAL A VERIFICAR:\n{message_text}\n\n"
            "ARTÍCULO DE REFERENCIA:\n{retrieved_article}\n\n"
            "Basándote ÚNICAMENTE en el artículo de referencia, escribe un veredicto de "
            "3 a 4 líneas explicando si el mensaje viral es verdadero o falso y por qué. "
            "Sé conciso, directo y cita el artículo. "
            "Escribe en párrafo continuo, sin viñetas ni listas.\n\n"
            "VEREDICTO:"
        ),
    )

    llm = Ollama(base_url=ollama_url, model=model_name, temperature=0.0)  # type: ignore[call-arg]
    chain = prompt | llm
    verdict_text: str = await chain.ainvoke(
        {"message_text": text, "retrieved_article": retrieved_article}
    )
    return verdict_text.strip()


# ── Private helpers ───────────────────────────────────────────────────────────

async def _search_qdrant(entities: list[str]) -> list[dict]:
    """Dense vector search against the local Qdrant 'fact_checks' collection."""
    if not entities:
        return []

    query_text = " ".join(entities)

    # embed() is CPU-bound; run in a thread to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    vectors: list[list[float]] = await loop.run_in_executor(None, embed, [query_text])
    if not vectors:
        return []

    qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection = os.getenv("QDRANT_COLLECTION", "fact_checks")

    try:
        client = AsyncQdrantClient(host=qdrant_host, port=qdrant_port)
        response = await client.query_points(
            collection_name=collection,
            query=vectors[0],
            limit=5,
            with_payload=True,
            )
        hits = response.points
    except Exception as exc:  # noqa: BLE001
        print(f"[rag] Qdrant search failed: {exc}")
        return []

    results = []
    for hit in hits:
        payload = hit.payload or {}
        article_text = f"{payload.get('title', '')} {payload.get('summary', '')}".strip()
        results.append({
            "score": hit.score,
            "url": payload.get("url", ""),
            "verdict": _normalise_rating(article_text),
            "text": article_text,
        })
    return results


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
