"""
Reconocimiento de entidades nombradas con SpaCy y selección dinámica de modelo de idioma.
Extrae conceptos clave de mensajes virales antes de la recuperación RAG.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import spacy

logger = logging.getLogger("verum.ner")

# Idioma soportado → nombre del modelo SpaCy
_MODELS: dict[str, str] = {
    "es": "es_core_news_lg",
    "en": "en_core_web_sm",
}

# Singletons de carga perezosa, uno por idioma detectado
_nlp_models: dict[str, Any] = {}


def _detect_language(text: str) -> str:
    """Devuelve el código ISO 639-1 del idioma de *text*, por defecto 'es' si falla."""
    try:
        from langdetect import detect, LangDetectException  # type: ignore[import]
        try:
            return detect(text)
        except LangDetectException:
            return "es"
    except ImportError:
        return "es"


def _get_nlp(lang: str = "es") -> Any:
    """Devuelve (y carga perezosamente) el modelo SpaCy para *lang*."""
    model_name = _MODELS.get(lang, "es_core_news_lg")
    if lang not in _nlp_models:
        logger.debug("[ner] Loading SpaCy model %r for lang=%r", model_name, lang)
        _nlp_models[lang] = spacy.load(model_name)
    return _nlp_models[lang]


def extract_entities(text: str) -> list[str]:
    """
    Limpia el texto y extrae entidades nombradas + noun chunks clave.
    Devuelve una lista deduplicada de cadenas de entidades para consultas RAG.
    """
    clean = re.sub(r"http\S+", "", text)
    clean = re.sub(r"[^\w\s]", " ", clean)

    lang = _detect_language(clean)
    logger.debug("[ner] Detected language: %s", lang)
    doc = _get_nlp(lang)(clean)

    # Named entities
    ner_entities: list[str] = [ent.text.strip() for ent in doc.ents]
    ner_lower: set[str] = {e.lower() for e in ner_entities}

    filtered_chunks: list[str] = []
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        # Mínimo 3 caracteres y con head que no sea stopword
        if len(chunk_text) < 3 or chunk.root.is_stop:
            continue
        chunk_lower = chunk_text.lower()
        # Omitir si ya está cubierto por (o cubre) una entidad nombrada
        if any(chunk_lower in ne or ne in chunk_lower for ne in ner_lower):
            continue
        filtered_chunks.append(chunk_text)

    # Ordenar chunks por longitud descendente (más informativos primero)
    filtered_chunks.sort(key=len, reverse=True)

    # Combinar: NER primero, luego chunks, máximo 8
    MAX_ENTITIES = 8
    combined = ner_entities + filtered_chunks
    result = list(dict.fromkeys(e for e in combined if len(e) > 2))[:MAX_ENTITIES]

    logger.info("[ner] Extracted %d entities: %s", len(result), result)
    return result


def is_gibberish(text: str) -> bool:
    """
    Devuelve True si *text* parece caracteres aleatorios o entrada incoherente
    que provocaría que el pipeline RAG alucinase un veredicto.
    Usa ratio alfa, longitud media de token, conteo de tokens significativos y
    ratio POS=X de SpaCy - al menos dos señales deben activarse simultáneamente.
    """
    clean = re.sub(r"http\S+", "", text)
    clean = re.sub(r"[^\w\s]", " ", clean).strip()

    if not clean:
        return True

    # Comprobaciones rápidas (sin SpaCy)
    if len(clean) < 3:
        logger.debug("[ner] Gibberish fast-path triggered for text=%.20r", text)
        return True
    alnum_ratio = sum(c.isalnum() for c in clean) / len(clean)
    if alnum_ratio < 0.3:
        logger.debug("[ner] Gibberish fast-path triggered for text=%.20r", text)
        return True

    # Comprobación rápida de vocales
    _VOWELS = set("aeiouáéíóúü")
    ws_alpha = [t for t in clean.split() if t.isalpha()]
    if ws_alpha:
        no_vowel_count = sum(1 for t in ws_alpha if not _VOWELS.intersection(t.lower()))
        # Con >= 5 tokens hay suficiente señal estadística; los patrones de
        # teclado aleatorio (p.ej. "asdf qwer zxcv") tienen ~40% de tokens sin vocal.
        threshold = 0.40 if len(ws_alpha) >= 5 else 0.50
        if no_vowel_count / len(ws_alpha) >= threshold:
            logger.debug("[ner] Gibberish vowel fast-path triggered for text=%.20r", text)
            return True

    lang = _detect_language(clean)
    logger.debug("[ner] is_gibberish lang: %s", lang)
    doc = _get_nlp(lang)(clean)

    if not doc.has_annotation("TAG"):
        return False  # el modelo no pudo parsear - dejar que el pipeline lo gestione

    total = max(len(doc), 1)
    alpha_tokens = [t for t in doc if t.is_alpha]
    alpha_count = max(len(alpha_tokens), 1)

    # Ratio OOV - solo significativo cuando el modelo tiene vectores (es_core_news_lg sí).
    if doc.vocab.vectors_length > 0:
        long_alpha = [t for t in alpha_tokens if len(t.text) > 2]
        if len(long_alpha) >= 4:
            oov_count = sum(1 for t in long_alpha if t.is_oov)
            oov_ratio = oov_count / len(long_alpha)
            if oov_ratio >= 0.70:
                logger.debug("[ner] Gibberish OOV fast-path: oov_ratio=%.2f", oov_ratio)
                return True

    # Ratio de racimos de consonantes (pseudo-acrónimo / golpeteo aleatorio)
    # ñ se trata como consonante en español; el patrón es independiente del idioma.
    _CONSONANT_CLUSTER = re.compile(r"[bcdfghjklmnñpqrstvwxyz]{3,}", re.IGNORECASE)
    if ws_alpha:
        cluster_count = sum(1 for t in ws_alpha if _CONSONANT_CLUSTER.search(t))
        consonant_ratio = cluster_count / len(ws_alpha)
        if consonant_ratio > 0.60:
            logger.debug(
                "[ner] Gibberish consonant fast-path: consonant_ratio=%.2f", consonant_ratio
            )
            return True

    alpha_ratio = len(alpha_tokens) / total
    avg_len = sum(len(t.text) for t in alpha_tokens) / alpha_count
    unknown_ratio = sum(1 for t in doc if t.pos_ == "X") / total
    meaningful_tokens = [t for t in doc if t.is_alpha and not t.is_stop and len(t.text) > 2]

    conditions = [
        alpha_ratio < 0.40,
        avg_len < 3.0,
        len(meaningful_tokens) < 4,
        unknown_ratio > 0.60,
    ]
    # Decisive combo: very few meaningful tokens AND very short avg length
    if len(meaningful_tokens) < 4 and avg_len < 4.5:
        return True
    return sum(conditions) >= 2
