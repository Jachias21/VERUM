"""
RAG pipeline — Hybrid Search (Local Qdrant → Google Fact Check API fallback)
followed by LLM synthesis via Ollama.

Architecture:
  Level 1: Hybrid vector search (dense + BM25 sparse, fused with RRF) against
           local Qdrant knowledge base.
  Level 2: On-demand Google Fact Check Tools API (if Level 1 confidence < threshold).
  Synthesis: Ollama SLM (Llama 3.2 / Qwen 2.5) generates a 3-line verdict.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid

import httpx
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector
from shared.schemas import NLPResult
from models.nlp.embeddings import embed, sparse_embed

logger = logging.getLogger("verum.rag")


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
    min_relevance = float(os.getenv("NLP_MIN_ARTICLE_SCORE", 0.40))

    logger.info(
        "hybrid_search: query_id=%s, entities=%s",
        query_id, entities,
    )

    # ── Early exit: no entities and text too short to search meaningfully ─────
    if not entities and len(text) < 20:
        logger.warning("hybrid_search: early-exit — text too short and no entities for query_id=%s", query_id)
        return NLPResult(
            query_id=query_id,
            extracted_entities=[],
            fact_check_matches=0,
            verdict="UNVERIFIED",
            summary=(
                "El texto es demasiado corto o no contiene entidades detectables. "
                "Envíame una afirmación o noticia completa para poder verificarla."
            ),
        )

    # ── Level 1: Local Qdrant hybrid search ──────────────────────────────────
    query_terms = entities if entities else [text[:200]]  # graceful degradation
    local_hits = await _search_qdrant(query_terms)

    top_score = local_hits[0]["score"] if local_hits else 0.0
    logger.info(
        "Qdrant returned %d hits, top_score=%.3f for query_id=%s",
        len(local_hits), top_score, query_id,
    )

    if local_hits and local_hits[0]["score"] >= threshold:
        best = local_hits[0]
        # Discard if article body or title is empty (badly formatted entry)
        if not best.get("text", "").strip():
            logger.info(
                "[rag] Discarding L1 hit (empty article body) score=%.3f for query_id=%s",
                best["score"], query_id,
            )
        else:
            return NLPResult(
                query_id=query_id,
                extracted_entities=entities,
                fact_check_matches=len(local_hits),
                source_url=best.get("url"),
                verdict=best.get("verdict", "UNVERIFIED"),
                summary=best.get("text", ""),  # article text passed to synthesize_verdict for LLM inference
            )

    # ── Level 2: parallel fallback (Google Fact Check + GNews) ─────────────────
    l2_query = " ".join(entities) if entities else text[:200]
    l1_score = local_hits[0]["score"] if local_hits else 0.0

    logger.warning(
        "L1 miss (score=%.3f, threshold=%.2f) — activating L2 fallback for query_id=%s",
        l1_score, threshold, query_id,
    )
    google_hits, gnews_hits = await asyncio.gather(
        _search_google_fact_check(l2_query),
        _search_gnews(l2_query),
    )
    logger.info(
        "L2 results — Google FC returned %d claims, GNews returned %d articles for query_id=%s",
        len(google_hits), len(gnews_hits), query_id,
    )

    if google_hits:
        best = google_hits[0]
        return NLPResult(
            query_id=query_id,
            extracted_entities=entities,
            fact_check_matches=len(google_hits),
            source_url=best.get("url"),
            verdict=best.get("verdict", "UNVERIFIED"),
            summary=best.get("text", ""),
        )

    if gnews_hits:
        best = gnews_hits[0]
        return NLPResult(
            query_id=query_id,
            extracted_entities=entities,
            fact_check_matches=len(gnews_hits),
            source_url=best.get("url"),
            verdict=best.get("verdict", "UNVERIFIED"),
            summary=best.get("text", ""),
        )

    # L2 also empty — use best local hit only if relevance score is acceptable
    if local_hits:
        best = local_hits[0]
        score = best["score"]
        if score >= min_relevance:
            return NLPResult(
                query_id=query_id,
                extracted_entities=entities,
                fact_check_matches=len(local_hits),
                source_url=best.get("url"),
                verdict=best.get("verdict", "UNVERIFIED"),
                summary=best.get("text", ""),
            )
        logger.info(
            "[rag] Discarding local hit score=%.3f < MIN_RELEVANCE=%.2f for query_id=%s",
            score, min_relevance, query_id,
        )

    return NLPResult(
        query_id=query_id,
        extracted_entities=entities,
        fact_check_matches=0,
        verdict="UNVERIFIED",
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

    _ollama_host = os.getenv("OLLAMA_HOST", "ollama")
    _ollama_port = os.getenv("OLLAMA_PORT", "11434")
    ollama_url = f"http://{_ollama_host}:{_ollama_port}"
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
            "Si el artículo de referencia describe claramente un bulo, fraude o contenido "
            "falso, el veredicto debe comenzar con 'VEREDICTO: FALSO'. "
            "Si confirma la veracidad, debe comenzar con 'VEREDICTO: VERDADERO'. "
            "Si no hay suficiente información, usar 'VEREDICTO: NO VERIFICADO'.\n\n"
            "VEREDICTO:"
        ),
    )

    logger.info("[rag] Ollama synthesis — model=%s url=%s", model_name, ollama_url)
    llm = OllamaLLM(base_url=ollama_url, model=model_name, temperature=0.0)  # type: ignore[call-arg]
    chain = prompt | llm

    MAX_RETRIES = 2
    OLLAMA_TIMEOUT_SECONDS = 45

    for attempt in range(MAX_RETRIES + 1):
        try:
            t0 = time.monotonic()
            verdict_text: str = await asyncio.wait_for(
                chain.ainvoke({"message_text": text, "retrieved_article": retrieved_article}),
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            verdict_stripped = verdict_text.strip()
            logger.info(
                "Ollama responded in %dms (attempt=%d): %r",
                elapsed_ms, attempt + 1, verdict_stripped[:150],
            )
            return verdict_stripped
        except asyncio.TimeoutError:
            logger.warning(
                "[rag] Ollama timeout after %ds (attempt=%d/%d)",
                OLLAMA_TIMEOUT_SECONDS, attempt + 1, MAX_RETRIES + 1,
            )
            if attempt == MAX_RETRIES:
                return (
                    f"Veredicto: {result.verdict}\n"
                    f"[El modelo de síntesis tardó demasiado. Fuente: {result.source_url or 'N/A'}]"
                )
            await asyncio.sleep(2 ** attempt)
        except Exception as exc:
            logger.error("[rag] Ollama synthesis error (attempt %d/%d): %s", attempt + 1, MAX_RETRIES + 1, exc)
            if attempt == MAX_RETRIES:
                return f"Veredicto: {result.verdict}\nFuente: {result.source_url or 'N/A'}"
            await asyncio.sleep(2 ** attempt)
    # Unreachable, but satisfies type checkers
    return f"Veredicto: {result.verdict}\nFuente: {result.source_url or 'N/A'}"


def _extract_verdict_from_llm_output(text: str) -> str | None:
    """
    Parse free-form LLM output and return a canonical verdict string.
    Returns "FAKE", "REAL", or "UNVERIFIED", or None if nothing matches.
    Checks Spanish labels first (VEREDICTO:), then English (VERDICT:).
    """
    if re.search(r"veredicto:\s*no[\s_]verificado", text, re.IGNORECASE):
        return "UNVERIFIED"
    if re.search(r"veredicto:\s*falso", text, re.IGNORECASE):
        return "FAKE"
    if re.search(r"veredicto:\s*verdadero", text, re.IGNORECASE):
        return "REAL"
    if re.search(r"verdict:\s*unverified", text, re.IGNORECASE):
        return "UNVERIFIED"
    if re.search(r"verdict:\s*(fake|false)", text, re.IGNORECASE):
        return "FAKE"
    if re.search(r"verdict:\s*(true|real)", text, re.IGNORECASE):
        return "REAL"
    return None


# ── Private helpers ───────────────────────────────────────────────────────────

async def _search_qdrant(entities: list[str]) -> list[dict]:
    """Hybrid vector search (dense + BM25 sparse, fused with RRF) against 'fact_checks'."""
    if not entities:
        return []

    query_text = " ".join(entities)

    # Both embeddings are CPU-bound; run together in a thread pool
    loop = asyncio.get_event_loop()

    def _embed_both(text: str):
        dense = embed([text])[0]                     # list[float]
        sparse_results = sparse_embed([text])        # list[SparseEmbedding]
        return dense, sparse_results[0]

    dense_vec, sparse_emb = await loop.run_in_executor(None, _embed_both, query_text)

    sparse_vec = SparseVector(
        indices=sparse_emb.indices.tolist(),
        values=sparse_emb.values.tolist(),
    )

    qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection = os.getenv("QDRANT_COLLECTION", "fact_checks")

    try:
        client = AsyncQdrantClient(host=qdrant_host, port=qdrant_port)
        response = await client.query_points(
            collection_name=collection,
            prefetch=[
                Prefetch(query=dense_vec, using="dense", limit=20),
                Prefetch(query=sparse_vec, using="sparse", limit=20),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=5,
            with_payload=True,
        )
        hits = response.points
    except Exception as exc:  # noqa: BLE001
        logger.error("[rag] Qdrant hybrid search failed: %s", exc)
        return []

    results = []
    for hit in hits:
        payload = hit.payload or {}
        article_text = f"{payload.get('title', '')} {payload.get('summary', '')}".strip()
        stored_verdict = payload.get("verdict", "UNVERIFIED")
        # Re-infer from article text when the stored verdict carries no signal
        if stored_verdict in ("UNVERIFIED", "", None):
            stored_verdict = _normalise_rating(article_text)
        results.append({
            "score": hit.score,
            "url": payload.get("url", ""),
            "verdict": stored_verdict,
            "text": article_text,
        })
    top_score = results[0]["score"] if results else 0.0
    logger.info("[rag] Qdrant returned %d hits, top_score=%.3f", len(results), top_score)
    return results


def _parse_google_claims(claims: list) -> list[dict]:
    results = []
    for claim in claims:
        review = claim.get("claimReview", [{}])[0]
        claim_text = claim.get("text", "")
        rating_text = review.get("textualRating", "")
        results.append({
            "score": 1.0,  # API results always treated as high confidence
            "verdict": _normalise_rating(rating_text + " " + claim_text),
            "url": review.get("url", ""),
            "text": f"{claim_text} — {rating_text}".strip(" —"),
        })
    return results


async def _search_google_fact_check(query: str) -> list[dict]:
    """Call Google Fact Check Tools API and normalise results.

    First attempts with languageCode=es; if that returns 0 claims, retries
    without languageCode to capture Latin-American fact-checks.
    """
    api_key = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "")
    if not api_key:
        logger.warning("[rag] Google FC API: key not configured — skipping")
        return []
    logger.info("[rag] Google FC API: key_present=True, querying...")
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                url, params={"query": query, "key": api_key, "languageCode": "es"}
            )
            if resp.status_code != 200:
                logger.warning("[rag] Google FC API returned HTTP %d", resp.status_code)
                return []
            claims = resp.json().get("claims", [])
            if claims:
                result = _parse_google_claims(claims)
                logger.info("[rag] Google FC returned %d claims for query=%.60r", len(result), query)
                return result
            # Retry without languageCode for broader coverage
            resp2 = await client.get(url, params={"query": query, "key": api_key})
            if resp2.status_code != 200:
                logger.warning("[rag] Google FC API (retry) returned HTTP %d", resp2.status_code)
                return []
            result = _parse_google_claims(resp2.json().get("claims", []))
            logger.info("[rag] Google FC (retry, no lang) returned %d claims for query=%.60r", len(result), query)
            return result
    except Exception as exc:
        logger.error("[rag] Google Fact Check API error: %s", exc)
        return []


async def _search_gnews(query: str) -> list[dict]:
    """Search GNews API for fact-check related articles.

    Returns [] silently when GNEWS_API_KEY is absent or the request fails.
    """
    api_key = os.getenv("GNEWS_API_KEY", "")
    if not api_key:
        logger.warning("[rag] GNews API: key not configured — skipping")
        return []
    logger.info("[rag] GNews API: key_present=True, querying...")
    url = f"https://gnews.io/api/v4/search?q={query}&lang=es&token={api_key}&max=5"
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
    except Exception as exc:
        logger.error("[rag] GNews API error: %s", exc)
        return []
    results = []
    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "") or ""
        combined = f"{title} {description}"
        results.append({
            "score": 0.7,
            "verdict": _normalise_rating(combined),
            "url": article.get("url", ""),
            "text": f"{title}. {description}".strip(". "),
        })
    logger.info("[rag] GNews returned %d articles for query=%.60r", len(results), query)
    return results


_FAKE_KEYWORDS = [
    "falso", "false", "fake", "bulo", "hoax", "incorrecto", "desinformación",
    "engaño", "manipulado", "mentira", "no es cierto", "sin evidencia",
    "misleading", "misinformation", "debunked", "desmentido", "incorrect",
]
_REAL_KEYWORDS = [
    "verdadero", "true", "correcto", "verified", "confirmado", "cierto",
    "demostrado", "correct",
]


def _normalise_rating(rating: str) -> str:
    rating_lower = rating.lower()
    if any(w in rating_lower for w in _FAKE_KEYWORDS):
        return "FAKE"
    if any(w in rating_lower for w in _REAL_KEYWORDS):
        return "REAL"
    return "UNVERIFIED"
