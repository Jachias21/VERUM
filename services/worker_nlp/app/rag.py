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
import unicodedata
import uuid

import httpx
from cachetools import TTLCache
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector
from shared.schemas import NLPResult
from models.nlp.embeddings import embed, sparse_embed

logger = logging.getLogger("verum.rag")

# In-memory cache for Google Fact Check API results (asyncio single-thread, no lock needed)
_google_fc_cache: TTLCache = TTLCache(maxsize=256, ttl=300)

# Session-level GNews circuit-breaker: set to True on first 403/429 to stop
# hammering the API when the key plan is exhausted or forbidden.
_gnews_session_disabled: bool = False


def _disable_gnews_session() -> None:
    global _gnews_session_disabled
    _gnews_session_disabled = True

# ── Ollama configuration ──────────────────────────────────────────────────────
OLLAMA_TIMEOUT_SECONDS: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600"))

# ── NL verdict patterns — compiled once at import time ────────────────────────
# Tier 2: Spanish natural-language signals
_FAKE_NL_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(es|son|resulta)\s+(un\s+)?(bulo|falso|falsa|mentira|fraude|fake|engaño)\b", re.IGNORECASE),
    re.compile(r"\bha\s+sido\s+desmentid[oa]\b", re.IGNORECASE),
    re.compile(r"\bno\s+es\s+cierto\b", re.IGNORECASE),
    re.compile(r"\bes\s+un\s+(bulo|hoax)\b", re.IGNORECASE),
]
_REAL_NL_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(es|son|resulta)\s+(verdader[oa]|ciert[oa]|real|correct[oa])\b", re.IGNORECASE),
    re.compile(r"\bse\s+confirma\b", re.IGNORECASE),
    re.compile(r"\bha\s+sido\s+confirmad[oa]\b", re.IGNORECASE),
]
_UNVERIFIED_NL_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bno\s+(se\s+puede|hay\s+forma\s+de|hay\s+manera\s+de)\s+confirmar\b", re.IGNORECASE),
    re.compile(r"\bno\s+hay\s+(suficiente\s+)?(información|evidencia)\b", re.IGNORECASE),
    re.compile(r"\bno\s+menciona\b", re.IGNORECASE),
]
# Tier 3: English natural-language signals
_FAKE_EN_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(is|are|appears?\s+to\s+be)\s+(fake|false|misinformation|a\s+hoax)\b", re.IGNORECASE),
    re.compile(r"\bhas\s+been\s+debunked\b", re.IGNORECASE),
    re.compile(r"\bis\s+not\s+true\b", re.IGNORECASE),
]
_REAL_EN_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(is|are)\s+(true|real|accurate|correct)\b", re.IGNORECASE),
    re.compile(r"\bhas\s+been\s+confirmed\b", re.IGNORECASE),
]
_UNVERIFIED_EN_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bcannot\s+be\s+confirmed\b", re.IGNORECASE),
    re.compile(r"\bnot\s+enough\s+(information|evidence)\b", re.IGNORECASE),
    re.compile(r"\bdoes\s+not\s+mention\b", re.IGNORECASE),
]


# ── LLM prompt templates ─────────────────────────────────────────────────────

_PROMPT_WITH_CONTEXT = """Eres un verificador de hechos. Tu tarea es decidir si un mensaje viral es VERDADERO, FALSO o NO VERIFICADO basándote en un artículo de referencia.

REGLAS:
1. El veredicto se aplica al MENSAJE VIRAL, no al artículo.
2. Si el artículo CONFIRMA lo que dice el mensaje → VERDADERO.
3. Si el artículo REFUTA lo que dice el mensaje (incluso si el artículo es un fact-check que desmiente una versión opuesta) → FALSO.
4. Si el artículo trata un tema relacionado pero NO determina la veracidad del mensaje → NO VERIFICADO.
5. Atención a las dobles negaciones: si el mensaje dice "X no es cierto" y el artículo dice "es falso que X" → ambos coinciden → VERDADERO.

EJEMPLOS:

Mensaje: "WhatsApp pasará a ser de pago"
Artículo: "Es falso que WhatsApp vaya a cobrar. La empresa lo ha desmentido."
Análisis: el artículo refuta el mensaje.
VEREDICTO: FALSO

Mensaje: "Las vacunas COVID son seguras y eficaces"
Artículo: "La OMS confirma la seguridad de las vacunas COVID tras estudios de fase 3."
Análisis: el artículo confirma el mensaje.
VEREDICTO: VERDADERO

Mensaje: "No hay nanopartículas en las personas vacunadas"
Artículo: "Es falso que un estudio japonés haya encontrado nanopartículas de ARNm en vacunados."
Análisis: el artículo desmiente la versión opuesta → confirma el mensaje.
VEREDICTO: VERDADERO

Mensaje: "El Gobierno regala 200€ a todos los pensionistas"
Artículo: "Análisis semanal de bulos en redes sociales: contenido tangencial."
Análisis: tema relacionado pero no determina la afirmación.
VEREDICTO: NO VERIFICADO

AHORA EVALÚA:

Mensaje viral: {claim}

Artículo de referencia: {context}

Tu primera línea DEBE empezar con "VEREDICTO: " seguido de FALSO, VERDADERO o NO VERIFICADO. Tras el veredicto, escribe 2-3 líneas explicando brevemente la decisión, citando el artículo.
"""

_PROMPT_GENERAL_KNOWLEDGE = """Eres un verificador de hechos experto. No tienes un artículo de referencia: usa tu conocimiento general para evaluar un mensaje viral.

REGLA DE ORO: ante CUALQUIER duda, responde NO VERIFICADO. Esta etiqueta NO es una derrota; es la respuesta CORRECTA cuando no puedes probar ni refutar el mensaje. Es preferible un NO VERIFICADO honesto a un FALSO precipitado.

CRITERIOS:

1. Responde FALSO solo cuando el mensaje contradice DIRECTAMENTE un hecho científico, histórico, geográfico o institucional bien establecido y de consenso amplio.
   Ejemplo: "La Tierra es plana", "El hombre no llegó a la Luna", "Las vacunas causan autismo".

2. Responde VERDADERO solo cuando el mensaje afirma un hecho científico, histórico, geográfico o institucional bien establecido y de consenso amplio.
   Ejemplo: "El Muro de Berlín cayó en 1989", "España es miembro de la UE", "El agua hierve a 100°C al nivel del mar".

3. Responde NO VERIFICADO en TODOS los demás casos, especialmente:
   - Noticias sobre eventos recientes (últimos 24 meses) que requieran información actualizada.
   - Declaraciones sobre personas vivas, políticos en activo, empresas, decisiones gubernamentales recientes.
   - Datos económicos, estadísticas, indicadores que cambian con el tiempo.
   - Detalles específicos médicos, legales, financieros que requieren verificación profesional.
   - Predicciones futuras, opiniones, valoraciones subjetivas.
   - Información parcialmente cierta o cierta solo bajo ciertas condiciones.
   - Cualquier afirmación donde no estés seguro al 95%.

EJEMPLOS DETALLADOS:

Mensaje: "El presidente del Gobierno firmará la ley X la próxima semana"
Análisis: predicción futura sobre actor político actual. No es verificable.
VEREDICTO: NO VERIFICADO

Mensaje: "El IPC de España en abril fue del 2,3%"
Análisis: dato económico específico y reciente. Requiere consulta al INE; no puedo confirmar de memoria.
VEREDICTO: NO VERIFICADO

Mensaje: "Pedro Sánchez es presidente del Gobierno de España"
Análisis: hecho político actual, pero requiere comprobar si sigue siéndolo hoy mismo.
VEREDICTO: NO VERIFICADO

Mensaje: "El cambio climático está causado principalmente por la actividad humana"
Análisis: consenso científico amplio respaldado por el IPCC.
VEREDICTO: VERDADERO

Mensaje: "Las vacunas contienen microchips para rastrear a la población"
Análisis: contradice directamente el conocimiento científico y técnico establecido.
VEREDICTO: FALSO

Mensaje: "El gobierno regala 200€ a todos los pensionistas este mes"
Análisis: requiere verificación documental específica que no tengo. No puedo confirmar ni desmentir.
VEREDICTO: NO VERIFICADO

AHORA EVALÚA:

Mensaje viral: {claim}

Tu primera línea DEBE empezar con "VEREDICTO: " seguido EXACTAMENTE de una de estas tres palabras: FALSO, VERDADERO o NO VERIFICADO. Tras el veredicto, escribe 2-3 líneas explicando tu razonamiento. Si no estás completamente seguro, di NO VERIFICADO.
"""


async def hybrid_search(
    query_id: uuid.UUID,
    text: str,
    entities: list[str],
) -> NLPResult:
    """
    Execute Level-1 local search; fall back to Level-2 API if needed.
    Returns a partially-filled NLPResult (verdict is placeholder; synthesize_verdict decides).
    """
    threshold = float(os.getenv("NLP_CONFIDENCE_THRESHOLD", 0.65))
    min_relevance = float(os.getenv("NLP_MIN_ARTICLE_SCORE", 0.35))
    min_overlap = float(os.getenv("NLP_TOPIC_OVERLAP_MIN", 0.30))

    logger.info(
        "hybrid_search: query_id=%s, entities=%s",
        query_id, entities,
    )

    # ── Early exit: no entities and text too short to search meaningfully ─────
    _min_len = int(os.getenv("NLP_MIN_TEXT_LENGTH", 50))
    if not entities and len(text) < _min_len:
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
        overlap = _topic_overlap_score(entities, best.get("text", ""))
        if not best.get("text", "").strip() or overlap < min_overlap:
            logger.info(
                "[rag] L1 hit discarded (empty body or overlap %.2f < %.2f) for query_id=%s",
                overlap, min_overlap, query_id,
            )
            # fall through to L2
        else:
            logger.info("[rag] L1 hit accepted (score=%.3f, overlap=%.2f) for query_id=%s",
                        best["score"], overlap, query_id)
            return NLPResult(
                query_id=query_id,
                extracted_entities=entities,
                fact_check_matches=len(local_hits),
                source_url=best.get("url"),
                verdict="UNVERIFIED",  # placeholder — LLM decides
                retrieved_context=best.get("text", ""),
                summary="",
            )

    # ── Level 2: parallel fallback (Google Fact Check + GNews) ─────────────────
    l2_query = _build_l2_query(entities, text)
    l1_score = local_hits[0]["score"] if local_hits else 0.0

    logger.warning(
        "L1 miss (score=%.3f, threshold=%.2f) — activating L2 fallback for query_id=%s",
        l1_score, threshold, query_id,
    )
    try:
        from langdetect import detect
        _lang = detect(text)
        if _lang not in ("es", "en"):
            _lang = "en"  # default to English for maximum international coverage
    except Exception:
        _lang = "es"
    logger.info("[rag] L2 querying with lang=%s for query_id=%s", _lang, query_id)
    google_hits, gnews_hits = await asyncio.gather(
        _search_google_fact_check(l2_query, lang=_lang),
        _search_gnews(l2_query, lang=_lang),
    )
    logger.info(
        "L2 results — Google FC returned %d claims, GNews returned %d articles for query_id=%s",
        len(google_hits), len(gnews_hits), query_id,
    )

    if google_hits:
        best = google_hits[0]
        overlap = _topic_overlap_score(entities, best.get("text", ""))
        if best.get("text", "").strip() and overlap >= min_overlap:
            logger.info(
                "[rag] L2 Google FC hit accepted (overlap=%.2f) for query_id=%s",
                overlap, query_id,
            )
            return NLPResult(
                query_id=query_id,
                extracted_entities=entities,
                fact_check_matches=len(google_hits),
                source_url=best.get("url"),
                verdict="UNVERIFIED",  # placeholder — LLM decides
                retrieved_context=best.get("text", ""),
                summary="",
            )
        logger.warning(
            "[rag] L2 Google FC hit discarded (empty body or overlap %.2f < %.2f) for query_id=%s",
            overlap, min_overlap, query_id,
        )

    if gnews_hits:
        best = gnews_hits[0]
        overlap = _topic_overlap_score(entities, best.get("text", ""))
        if best.get("text", "").strip() and overlap >= min_overlap:
            logger.info(
                "[rag] L2 GNews hit accepted (overlap=%.2f) for query_id=%s",
                overlap, query_id,
            )
            return NLPResult(
                query_id=query_id,
                extracted_entities=entities,
                fact_check_matches=len(gnews_hits),
                source_url=best.get("url"),
                verdict="UNVERIFIED",  # placeholder — LLM decides
                retrieved_context=best.get("text", ""),
                summary="",
            )
        logger.warning(
            "[rag] L2 GNews hit discarded (empty body or overlap %.2f < %.2f) for query_id=%s",
            overlap, min_overlap, query_id,
        )

    # ── L1 medium: use best local hit if score >= min_relevance ──────────────
    if local_hits:
        best = local_hits[0]
        score = best["score"]
        if score >= min_relevance:
            overlap = _topic_overlap_score(entities, best.get("text", ""))
            if best.get("text", "").strip() and overlap >= min_overlap:
                logger.info(
                    "[rag] L1 medium hit accepted (score=%.3f, overlap=%.2f) for query_id=%s",
                    score, overlap, query_id,
                )
                return NLPResult(
                    query_id=query_id,
                    extracted_entities=entities,
                    fact_check_matches=len(local_hits),
                    source_url=best.get("url"),
                    verdict="UNVERIFIED",  # placeholder — LLM decides
                    retrieved_context=best.get("text", ""),
                    summary="",
                )
            logger.info(
                "[rag] L1 medium hit discarded (overlap %.2f < %.2f or empty body) for query_id=%s",
                overlap, min_overlap, query_id,
            )
        else:
            logger.info(
                "[rag] Discarding local hit score=%.3f < MIN_RELEVANCE=%.2f for query_id=%s",
                score, min_relevance, query_id,
            )

    # ── L3: no reliable context — LLM uses general knowledge ─────────────────
    logger.info(
        "[rag] No reliable context found — invoking LLM with general knowledge for query_id=%s",
        query_id,
    )
    return NLPResult(
        query_id=query_id,
        extracted_entities=entities,
        fact_check_matches=0,
        source_url=None,
        verdict="UNVERIFIED",
        retrieved_context="",
        summary="__USE_GENERAL_KNOWLEDGE__",  # sentinel for synthesize_verdict
    )


async def synthesize_verdict(result: NLPResult, claim_text: str) -> NLPResult:
    """
    Invoke LLM to decide the final verdict given retrieved context and the user claim.
    Detects the __USE_GENERAL_KNOWLEDGE__ sentinel to select the appropriate prompt.
    Sets result.verdict and result.summary in-place and returns the modified result.
    """
    use_general_knowledge = result.summary == "__USE_GENERAL_KNOWLEDGE__"

    if use_general_knowledge:
        prompt_template = _PROMPT_GENERAL_KNOWLEDGE
        prompt_inputs = {"claim": claim_text[:1500]}
        input_variables = ["claim"]
    else:
        prompt_template = _PROMPT_WITH_CONTEXT
        context_truncated = (result.retrieved_context or "")[:1500]
        prompt_inputs = {"claim": claim_text[:1500], "context": context_truncated}
        input_variables = ["claim", "context"]

    _ollama_host = os.getenv("OLLAMA_HOST", "ollama")
    _ollama_port = os.getenv("OLLAMA_PORT", "11434")
    ollama_url = f"http://{_ollama_host}:{_ollama_port}"
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:14b-instruct-q4_K_M")

    prompt = PromptTemplate(
        input_variables=input_variables,
        template=prompt_template,
    )

    logger.info(
        "[rag] Ollama synthesis — model=%s url=%s use_general_knowledge=%s",
        model_name, ollama_url, use_general_knowledge,
    )
    llm = OllamaLLM(base_url=ollama_url, model=model_name, temperature=0.0)  # type: ignore[call-arg]
    chain = prompt | llm

    MAX_RETRIES = 2

    for attempt in range(MAX_RETRIES + 1):
        try:
            t0 = time.monotonic()
            verdict_text: str = await asyncio.wait_for(
                chain.ainvoke(prompt_inputs),
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            verdict_stripped = verdict_text.strip()
            logger.info(
                "Ollama responded in %dms (attempt=%d): %r",
                elapsed_ms, attempt + 1, verdict_stripped[:150],
            )
            extracted_verdict = _extract_verdict_from_llm_output(verdict_stripped)
            if extracted_verdict is not None:
                result.verdict = extracted_verdict
            result.summary = verdict_stripped
            return result
        except asyncio.TimeoutError:
            logger.warning(
                "[rag] Ollama timeout after %ds (attempt=%d/%d)",
                OLLAMA_TIMEOUT_SECONDS, attempt + 1, MAX_RETRIES + 1,
            )
            if attempt == MAX_RETRIES:
                result.summary = (
                    "El modelo de síntesis tardó demasiado en responder. "
                    "Consulta directamente la fuente para verificar manualmente."
                )
                return result
            await asyncio.sleep(2 ** attempt)
        except Exception as exc:
            logger.error("[rag] Ollama synthesis error (attempt %d/%d): %s", attempt + 1, MAX_RETRIES + 1, exc)
            if attempt == MAX_RETRIES:
                result.summary = (
                    "No se ha podido generar el resumen explicativo. "
                    "Consulta directamente la fuente."
                )
                return result
            await asyncio.sleep(2 ** attempt)
    # Unreachable, but satisfies type checkers
    result.summary = "No se ha podido generar el resumen explicativo. Consulta directamente la fuente."
    return result


def _extract_verdict_from_llm_output(text: str) -> str | None:
    """
    Parse free-form LLM output and return a canonical verdict string.
    Returns "FAKE", "REAL", or "UNVERIFIED", or None if nothing matches.

    Priority:
      Tier 1 (strict)  — VEREDICTO:/VERDICT: prefix patterns (always wins).
      Tier 2 (NL ES)   — Spanish natural-language signals.
      Tier 3 (NL EN)   — English natural-language signals.

    Within Tier 2/3, UNVERIFIED beats FAKE (more conservative).
    """
    # ── Tier 1: strict prefix patterns ────────────────────────────────────────
    if re.search(r"veredicto:\s*no[\s_-]verificado", text, re.IGNORECASE):
        return "UNVERIFIED"
    if re.search(r"veredicto:\s*falso\b", text, re.IGNORECASE):
        return "FAKE"
    if re.search(r"veredicto:\s*verdadero\b", text, re.IGNORECASE):
        return "REAL"
    if re.search(r"verdict:\s*unverified", text, re.IGNORECASE):
        return "UNVERIFIED"
    if re.search(r"verdict:\s*(fake|false)", text, re.IGNORECASE):
        return "FAKE"
    if re.search(r"verdict:\s*(true|real)", text, re.IGNORECASE):
        return "REAL"

    # ── Tier 2: Spanish natural-language signals ───────────────────────────────
    _t2_fake = any(p.search(text) for p in _FAKE_NL_PATTERNS)
    _t2_real = any(p.search(text) for p in _REAL_NL_PATTERNS)
    _t2_unverified = any(p.search(text) for p in _UNVERIFIED_NL_PATTERNS)

    if _t2_unverified:
        return "UNVERIFIED"
    if _t2_fake:
        return "FAKE"
    if _t2_real:
        return "REAL"

    # ── Tier 3: English natural-language signals ───────────────────────────────
    _t3_fake = any(p.search(text) for p in _FAKE_EN_PATTERNS)
    _t3_real = any(p.search(text) for p in _REAL_EN_PATTERNS)
    _t3_unverified = any(p.search(text) for p in _UNVERIFIED_EN_PATTERNS)

    if _t3_unverified:
        return "UNVERIFIED"
    if _t3_fake:
        return "FAKE"
    if _t3_real:
        return "REAL"

    return None


# ── Private helpers ───────────────────────────────────────────────────────────

def _topic_overlap_score(entities: list[str], article_text: str) -> float:
    """Return the fraction of *entities* that appear as substrings in *article_text*.

    Both sides are lowercased and accent-stripped before comparison so that
    "Ébola" matches "ebola".  Returns 0.0 when *entities* is empty.
    """
    if not entities:
        return 0.0

    def _normalize(s: str) -> str:
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()

    norm_article = _normalize(article_text)
    matches = sum(1 for e in entities if _normalize(e) in norm_article)
    return matches / len(entities)


async def _search_qdrant(entities: list[str]) -> list[dict]:
    """Hybrid vector search (dense + BM25 sparse, fused with RRF) against 'fact_checks'."""
    if not entities:
        return []

    query_text = " ".join(entities)
    logger.info("[rag] Qdrant query_text: %r (n_entities=%d)", query_text, len(entities))

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

    _MAX_RETRIES = 2
    last_exc: Exception | None = None
    hits = []
    for attempt in range(_MAX_RETRIES + 1):
        if attempt > 0:
            delay = 2 ** (attempt - 1)  # 1 s, 2 s
            logger.warning("[rag] Qdrant retry attempt %d/%d (sleeping %ds)", attempt, _MAX_RETRIES, delay)
            await asyncio.sleep(delay)
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
            break  # success
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("[rag] Qdrant hybrid search attempt %d failed: %s", attempt + 1, exc)
    else:
        logger.error("[rag] Qdrant unreachable after %d retries: %s", _MAX_RETRIES, last_exc)
        return []

    logger.info(
        "[rag] Qdrant raw scores: %s",
        [round(getattr(h, "score", 0.0), 4) for h in hits],
    )
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


async def _search_google_fact_check(query: str, lang: str = "es") -> list[dict]:
    """Call Google Fact Check Tools API and normalise results.

    First attempts with languageCode=<lang>; if that returns 0 claims, retries
    without languageCode to capture fact-checks in any language.
    Results are cached in memory for 5 minutes to avoid redundant API calls.
    """
    # Sanitise query before sending to Google FC
    query_clean = re.sub(r"\s+", " ", query).strip()
    if not query_clean:
        logger.warning("[rag] Google FC: empty query after sanitisation — skipping")
        return []
    if len(query_clean) > 150:
        query_clean = query_clean[:150].rsplit(" ", 1)[0]
    logger.info("[rag] Google FC query: %r", query_clean)

    cache_key = query_clean.lower()
    if cache_key in _google_fc_cache:
        logger.info("[rag] Google FC: cache hit for query=%.60r", query_clean)
        return _google_fc_cache[cache_key]

    api_key = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "")
    if not api_key:
        logger.warning("[rag] Google FC API: key not configured — skipping")
        return []
    logger.info("[rag] Google FC API: key_present=True, querying...")
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                url, params={"query": query_clean, "key": api_key, "languageCode": lang}
            )
            if resp.status_code != 200:
                logger.warning("[rag] Google FC API returned HTTP %d", resp.status_code)
                return []
            claims = resp.json().get("claims", [])
            if claims:
                result = _parse_google_claims(claims)
                logger.info("[rag] Google FC returned %d claims for query=%.60r", len(result), query_clean)
                _google_fc_cache[cache_key] = result
                return result
            # Retry without languageCode for broader coverage
            resp2 = await client.get(url, params={"query": query_clean, "key": api_key})
            if resp2.status_code != 200:
                logger.warning("[rag] Google FC API (retry) returned HTTP %d", resp2.status_code)
                return []
            result = _parse_google_claims(resp2.json().get("claims", []))
            logger.info("[rag] Google FC (retry, no lang) returned %d claims for query=%.60r", len(result), query_clean)
            _google_fc_cache[cache_key] = result
            return result
    except Exception as exc:
        logger.error("[rag] Google Fact Check API error: %s", exc)
        return []


async def _search_gnews(query: str, lang: str = "es") -> list[dict]:
    """Search GNews API for fact-check related articles.

    Returns [] silently when GNEWS_API_KEY is absent, the request fails,
    or the API has returned a 403/429 during this session.
    """
    api_key = os.getenv("GNEWS_API_KEY", "")
    if not api_key:
        logger.warning("[rag] GNews API: key not configured — skipping")
        return []
    if _gnews_session_disabled:
        logger.debug("[rag] GNews API: skipping — disabled for this session (prior 403/429)")
        return []
    logger.info("[rag] GNews API: key_present=True, querying...")
    url = "https://gnews.io/api/v4/search"
    # Sanitise query before sending to GNews
    query_clean = re.sub(r"\s+", " ", query).strip()
    if not query_clean:
        logger.warning("[rag] GNews: empty query after sanitisation — skipping")
        return []
    if len(query_clean) > 150:
        query_clean = query_clean[:150].rsplit(" ", 1)[0]
    logger.info("[rag] GNews query: %r", query_clean)
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(
                url,
                params={"q": query_clean, "lang": lang, "token": api_key, "max": 5},
            )
            if resp.status_code == 403:
                _disable_gnews_session()
                logger.warning(
                    "[rag] GNews API: 403 Forbidden — API key plan does not allow this request. "
                    "GNews will be skipped for the rest of this session."
                )
                return []
            if resp.status_code == 429:
                _disable_gnews_session()
                logger.warning(
                    "[rag] GNews API: 429 Too Many Requests — rate limit exceeded. "
                    "GNews will be skipped for the rest of this session."
                )
                return []
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
    except httpx.HTTPStatusError as exc:
        logger.warning("[rag] GNews API HTTP error %s: %s", exc.response.status_code, exc)
        return []
    except Exception as exc:
        logger.warning("[rag] GNews API error: %s", exc)
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


_LEADING_ARTICLES = (
    "la", "el", "los", "las", "un", "una", "unos", "unas",
    "the", "a", "an",
)


def _normalize_for_dedup(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s.lower()).strip()


def _build_l2_query(entities: list[str], fallback_text: str) -> str:
    """Build a concise, clean search query for L2 APIs from extracted entities.

    Filters empty/whitespace entities, entities beginning with a leading article,
    and long/noisy noun phrases (> 4 words or > 40 characters).  Deduplicates
    semantically (substring containment on normalised forms) and limits the
    result to 3 entities.  Falls back to the first 100 characters of the
    original text when no suitable entities are available.
    """
    if not entities:
        return re.sub(r"\s+", " ", fallback_text[:100]).strip()

    # 1. Drop empty / whitespace-only entities
    filtered = [e for e in entities if e.strip()]

    # 2. Drop entities whose first word is a leading article
    filtered = [
        e for e in filtered
        if e.strip().split()[0].lower() not in _LEADING_ARTICLES
    ]

    # 3. Keep only short, clean tokens: ≤ 4 words and ≤ 40 characters
    filtered = [e for e in filtered if len(e) <= 40 and len(e.split()) <= 4]

    # 4. Semantic deduplication: if norm(a) is a substring of norm(b), keep only b
    norms = [_normalize_for_dedup(e) for e in filtered]
    keep = [True] * len(filtered)
    for i in range(len(filtered)):
        for j in range(len(filtered)):
            if i == j or not keep[i]:
                continue
            # If norm[i] is contained in norm[j], i is the shorter/subset — drop it
            if norms[i] in norms[j] and norms[i] != norms[j]:
                keep[i] = False
    deduped = [e for e, k in zip(filtered, keep) if k]

    # 5. If all filters removed everything, fall back to original text
    if not deduped:
        return re.sub(r"\s+", " ", fallback_text[:100]).strip()

    # 6. Limit to 3 most prominent entities
    selected = deduped[:3]

    # 7. Collapse any double spaces in the final string
    return re.sub(r"\s+", " ", " ".join(selected)).strip()
