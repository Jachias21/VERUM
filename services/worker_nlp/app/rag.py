"""
Pipeline RAG - búsqueda híbrida en Qdrant (densa + BM25, fusión RRF) con
Google Fact Check y GNews como fallbacks, seguido de síntesis por LLM (Ollama).
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

# Caché en memoria para resultados de la API Google Fact Check (asyncio mono-hilo, sin lock)
_google_fc_cache: TTLCache = TTLCache(maxsize=256, ttl=300)

# Circuit-breaker de sesión para GNews: se activa en el primer 403/429 para
# dejar de llamar a la API cuando la cuota del plan está agotada.
_gnews_session_disabled: bool = False


def _disable_gnews_session() -> None:
    global _gnews_session_disabled
    _gnews_session_disabled = True

OLLAMA_TIMEOUT_SECONDS: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600"))
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
]
# English NL signals
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

CRITERIOS (en orden de aplicación):

1. Responde FALSO cuando el mensaje contradice un hecho científico, histórico, geográfico o institucional bien establecido. No hace falta consenso del 100%: basta con que la comunidad científica o las instituciones relevantes lo hayan refutado.
   Ejemplo: "La Tierra es plana", "El hombre no llegó a la Luna", "Las vacunas causan autismo", bulos virales conocidos sobre salud o política.

2. Responde VERDADERO cuando el mensaje afirma un hecho bien establecido y verificable.
   Ejemplo: "El Muro de Berlín cayó en 1989", "España es miembro de la UE", "El agua hierve a 100°C al nivel del mar".

3. Responde NO VERIFICADO solo cuando genuinamente no puedas clasificarlo como FALSO o VERDADERO:
   - Predicciones futuras.
   - Datos estadísticos muy específicos que cambian con el tiempo (IPC exacto, cifras de paro de un mes concreto).
   - Afirmaciones sobre decisiones gubernamentales muy recientes sin información establecida.
   - Opiniones subjetivas o valoraciones personales.

EJEMPLOS:

Mensaje: "El presidente del Gobierno firmará la ley X la próxima semana"
Análisis: predicción futura. No verificable.
VEREDICTO: NO VERIFICADO

Mensaje: "El cambio climático está causado principalmente por la actividad humana"
Análisis: consenso científico amplio (IPCC).
VEREDICTO: VERDADERO

Mensaje: "Las vacunas contienen microchips para rastrear a la población"
Análisis: bulo conocido que contradice el conocimiento científico y técnico.
VEREDICTO: FALSO

Mensaje: "El 5G causa cáncer"
Análisis: afirmación refutada por la OMS y organismos de salud.
VEREDICTO: FALSO

Mensaje: "El gobierno regala 200€ a todos los pensionistas este mes"
Análisis: dato administrativo específico y reciente. Sin información verificable.
VEREDICTO: NO VERIFICADO

AHORA EVALÚA:

Mensaje viral: {claim}

Tu primera línea DEBE empezar con "VEREDICTO: " seguido EXACTAMENTE de una de estas tres palabras: FALSO, VERDADERO o NO VERIFICADO. Tras el veredicto, escribe 2-3 líneas explicando tu razonamiento.
"""


async def hybrid_search(
    query_id: uuid.UUID,
    text: str,
    entities: list[str],
) -> NLPResult:
    """Búsqueda local en Qdrant con fallback a APIs externas si la confianza es baja.
    Devuelve un NLPResult parcialmente relleno; synthesize_verdict decide el veredicto final.
    """
    threshold = float(os.getenv("NLP_CONFIDENCE_THRESHOLD", 0.65))
    min_relevance = float(os.getenv("NLP_MIN_ARTICLE_SCORE", 0.35))
    min_overlap = float(os.getenv("NLP_TOPIC_OVERLAP_MIN", 0.30))

    logger.info(
        "hybrid_search: query_id=%s, entities=%s",
        query_id, entities,
    )

    # Salida temprana: texto demasiado corto y sin entidades
    _min_len = int(os.getenv("NLP_MIN_TEXT_LENGTH", 50))
    if not entities and len(text) < _min_len:
        logger.warning("hybrid_search: salida temprana - texto corto y sin entidades para query_id=%s", query_id)
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

    # Local Qdrant hybrid search
    query_terms = entities if entities else [text[:200]]  # degradación controlada
    local_hits = await _search_qdrant(query_terms)

    top_score = local_hits[0]["score"] if local_hits else 0.0
    logger.info(
        "Qdrant devolvió %d resultados, puntuación máxima=%.3f para query_id=%s",
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
                verdict="UNVERIFIED",  # marcador - el LLM decide
                retrieved_context=best.get("text", ""),
                summary="",
            )

    # Fallback L2: Google Fact Check + GNews
    l2_query = _build_l2_query(entities, text)
    l1_score = local_hits[0]["score"] if local_hits else 0.0

    logger.warning(
        "L1 miss (puntuación=%.3f, umbral=%.2f) - activando fallback L2 para query_id=%s",
        l1_score, threshold, query_id,
    )
    try:
        from langdetect import detect
        _lang = detect(text)
        if _lang not in ("es", "en"):
            _lang = "en"  # por defecto inglés para mayor cobertura internacional
    except Exception:
        _lang = "es"
    logger.info("[rag] L2 consultando con lang=%s para query_id=%s", _lang, query_id)
    google_hits, gnews_hits = await asyncio.gather(
        _search_google_fact_check(l2_query, lang=_lang),
        _search_gnews(l2_query, lang=_lang),
    )
    logger.info(
        "L2 - Google FC devolvió %d claims, GNews devolvió %d artículos para query_id=%s",
        len(google_hits), len(gnews_hits), query_id,
    )

    if google_hits:
        best = google_hits[0]
        overlap = _topic_overlap_score(entities, best.get("text", ""))
        if best.get("text", "").strip() and overlap >= min_overlap:
            logger.info(
                "[rag] L2 Google FC aceptado (overlap=%.2f) para query_id=%s",
                overlap, query_id,
            )
            return NLPResult(
                query_id=query_id,
                extracted_entities=entities,
                fact_check_matches=len(google_hits),
                source_url=best.get("url"),
                verdict="UNVERIFIED",  # marcador - el LLM decide
                retrieved_context=best.get("text", ""),
                summary="",
            )
        logger.warning(
            "[rag] L2 Google FC descartado (cuerpo vacío u overlap %.2f < %.2f) para query_id=%s",
            overlap, min_overlap, query_id,
        )

    if gnews_hits:
        best = gnews_hits[0]
        overlap = _topic_overlap_score(entities, best.get("text", ""))
        if best.get("text", "").strip() and overlap >= min_overlap:
            logger.info(
                "[rag] L2 GNews aceptado (overlap=%.2f) para query_id=%s",
                overlap, query_id,
            )
            return NLPResult(
                query_id=query_id,
                extracted_entities=entities,
                fact_check_matches=len(gnews_hits),
                source_url=best.get("url"),
                verdict="UNVERIFIED",  # marcador - el LLM decide
                retrieved_context=best.get("text", ""),
                summary="",
            )
        logger.warning(
            "[rag] L2 GNews descartado (cuerpo vacío u overlap %.2f < %.2f) para query_id=%s",
            overlap, min_overlap, query_id,
        )

    # L1 medium: use best local hit if score >= min_relevance
    if local_hits:
        best = local_hits[0]
        score = best["score"]
        if score >= min_relevance:
            overlap = _topic_overlap_score(entities, best.get("text", ""))
            if best.get("text", "").strip() and overlap >= min_overlap:
                logger.info(
                    "[rag] L1 medio aceptado (puntuación=%.3f, overlap=%.2f) para query_id=%s",
                    score, overlap, query_id,
                )
                return NLPResult(
                    query_id=query_id,
                    extracted_entities=entities,
                    fact_check_matches=len(local_hits),
                    source_url=best.get("url"),
                    verdict="UNVERIFIED",  # marcador - el LLM decide
                    retrieved_context=best.get("text", ""),
                    summary="",
                )
                logger.info(
                    "[rag] L1 medio descartado (overlap %.2f < %.2f o cuerpo vacío) para query_id=%s",
                    overlap, min_overlap, query_id,
                )
        else:
            logger.info(
                "[rag] Descartando resultado local puntuación=%.3f < MIN_RELEVANCE=%.2f para query_id=%s",
                score, min_relevance, query_id,
            )

    logger.info(
        "[rag] Sin contexto fiable - invocando LLM con conocimiento general para query_id=%s",
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
    Invoca el LLM para decidir el veredicto final dado el contexto recuperado y el claim.
    Detecta el centinela __USE_GENERAL_KNOWLEDGE__ para seleccionar el prompt adecuado.
    Actualiza result.verdict y result.summary en el mismo objeto y lo devuelve.
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
        "[rag] Síntesis Ollama - model=%s url=%s uso_conocimiento_general=%s",
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
                "Ollama respondió en %dms (intento=%d): %r",
                elapsed_ms, attempt + 1, verdict_stripped[:150],
            )
            extracted_verdict = _extract_verdict_from_llm_output(verdict_stripped)
            if extracted_verdict is not None:
                result.verdict = extracted_verdict
            result.summary = verdict_stripped
            return result
        except asyncio.TimeoutError:
            logger.warning(
                "[rag] Timeout Ollama tras %ds (intento=%d/%d)",
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
            logger.error("[rag] Error de síntesis Ollama (intento %d/%d): %s", attempt + 1, MAX_RETRIES + 1, exc)
            if attempt == MAX_RETRIES:
                result.summary = (
                    "No se ha podido generar el resumen explicativo. "
                    "Consulta directamente la fuente."
                )
                return result
            await asyncio.sleep(2 ** attempt)
    # Inalcanzable, satisface al comprobador de tipos
    result.summary = "No se ha podido generar el resumen explicativo. Consulta directamente la fuente."
    return result


def _extract_verdict_from_llm_output(text: str) -> str | None:
    """
    Parsea la salida libre del LLM y devuelve "FAKE", "REAL", "UNVERIFIED" o None.
    Los patrones de prefijo VEREDICTO:/VERDICT: tienen prioridad; las señales NL
    en español se comprueban antes que las inglesas. UNVERIFIED gana sobre FAKE en fallbacks NL.
    """
    # Patrones estrictos con prefijo VEREDICTO:/VERDICT:
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

    # Primera línea corta: el LLM puede escribir el veredicto solo, sin prefijo
    first_line = text.split("\n")[0].strip()
    if len(first_line) <= 25:
        fl_upper = first_line.upper()
        if fl_upper in ("NO VERIFICADO", "NO_VERIFICADO", "UNVERIFIED"):
            return "UNVERIFIED"
        if fl_upper in ("FALSO", "FALSE", "FAKE"):
            return "FAKE"
        if fl_upper in ("VERDADERO", "TRUE", "REAL"):
            return "REAL"

    # Fallback NL en español - FAKE/REAL tienen prioridad sobre UNVERIFIED
    _t2_fake = any(p.search(text) for p in _FAKE_NL_PATTERNS)
    _t2_real = any(p.search(text) for p in _REAL_NL_PATTERNS)
    _t2_unverified = any(p.search(text) for p in _UNVERIFIED_NL_PATTERNS)

    if _t2_fake and not _t2_real:
        return "FAKE"
    if _t2_real and not _t2_fake:
        return "REAL"
    if _t2_unverified:
        return "UNVERIFIED"

    # Fallback NL en inglés
    _t3_fake = any(p.search(text) for p in _FAKE_EN_PATTERNS)
    _t3_real = any(p.search(text) for p in _REAL_EN_PATTERNS)
    _t3_unverified = any(p.search(text) for p in _UNVERIFIED_EN_PATTERNS)

    if _t3_fake and not _t3_real:
        return "FAKE"
    if _t3_real and not _t3_fake:
        return "REAL"
    if _t3_unverified:
        return "UNVERIFIED"

    return None


_NO_INFO_PHRASES = (
    "no tengo información",
    "no menciona",
    "no puedo confirmar",
    "no hay información",
    "no se menciona",
    "sin información",
    # English equivalents
    "no information",
    "cannot confirm",
    "can't confirm",
    "no relevant",
)


def _is_no_info_response(text: str) -> bool:
    """Devuelve True si el resumen del LLM indica que no encontró información relevante."""
    lower = text.lower()
    return any(phrase in lower for phrase in _NO_INFO_PHRASES)


# Generic fallback message shown when the LLM had no verdict AND no relevant info.
NO_INFO_SUMMARY = (
    "No encontré información verificada sobre esto.\n\n"
    "No tenemos esta noticia en nuestra base de datos ni en fuentes de "
    "fact-checking. Esto no significa que sea falsa — simplemente no hay registros.\n\n"
    "💡 Comprueba en: maldita.es · newtral.es · snopes.com"
)


def resolve_final_verdict(result: NLPResult) -> NLPResult:
    """Decide el veredicto y resumen finales a partir de un NLPResult sintetizado por el LLM.
    Única fuente de verdad compartida por el worker y el script de evaluación.
    Extrae el veredicto de la salida del LLM; solo cae a UNVERIFIED cuando no hay
    veredicto claro Y el texto indica que no hay información relevante.
    Muta y devuelve ``result``.
    """
    llm_verdict = _extract_verdict_from_llm_output(result.summary)
    if llm_verdict is not None:
        result.verdict = llm_verdict
        return result

    if _is_no_info_response(result.summary):
        result.verdict = "UNVERIFIED"
        result.summary = NO_INFO_SUMMARY

    return result


def _topic_overlap_score(entities: list[str], article_text: str) -> float:
    """Devuelve la fracción de *entities* que aparecen como subcadenas en *article_text*.
    Ambos lados se normalizan (minúsculas y sin acentos) antes de comparar.
    Devuelve 0.0 cuando *entities* está vacío.
    """
    if not entities:
        return 0.0

    def _normalize(s: str) -> str:
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()

    norm_article = _normalize(article_text)
    matches = sum(1 for e in entities if _normalize(e) in norm_article)
    return matches / len(entities)


async def _search_qdrant(entities: list[str]) -> list[dict]:
    """Búsqueda vectorial híbrida (densa + BM25 sparse, fusionada con RRF) en 'fact_checks'."""
    if not entities:
        return []

    query_text = " ".join(entities)
    logger.info("[rag] Qdrant query_text: %r (n_entities=%d)", query_text, len(entities))

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
            delay = 2 ** (attempt - 1)
            logger.warning("[rag] Reintento Qdrant %d/%d (esperando %ds)", attempt, _MAX_RETRIES, delay)
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
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("[rag] Búsqueda híbrida Qdrant intento %d falló: %s", attempt + 1, exc)
    else:
        logger.error("[rag] Qdrant inaccesible tras %d reintentos: %s", _MAX_RETRIES, last_exc)
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
        logger.warning("[rag] Google FC: empty query after sanitisation - skipping")
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
        logger.warning("[rag] Google FC API: key not configured - skipping")
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
        logger.warning("[rag] GNews API: key not configured - skipping")
        return []
    if _gnews_session_disabled:
        logger.debug("[rag] GNews API: skipping - disabled for this session (prior 403/429)")
        return []
    logger.info("[rag] GNews API: key_present=True, querying...")
    url = "https://gnews.io/api/v4/search"
    # Sanitise query before sending to GNews
    query_clean = re.sub(r"\s+", " ", query).strip()
    if not query_clean:
        logger.warning("[rag] GNews: empty query after sanitisation - skipping")
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
                    "[rag] GNews API: 403 Forbidden - API key plan does not allow this request. "
                    "GNews will be skipped for the rest of this session."
                )
                return []
            if resp.status_code == 429:
                _disable_gnews_session()
                logger.warning(
                    "[rag] GNews API: 429 Too Many Requests - rate limit exceeded. "
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
            # If norm[i] is contained in norm[j], i is the shorter/subset - drop it
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
