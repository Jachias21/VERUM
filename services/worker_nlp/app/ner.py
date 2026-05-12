"""
Named Entity Recognition using SpaCy with dynamic language model selection.
Extracts key concepts from viral messages before RAG retrieval.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import spacy

logger = logging.getLogger("verum.ner")

# Supported language → SpaCy model name
_MODELS: dict[str, str] = {
    "es": "es_core_news_lg",
    "en": "en_core_web_sm",
}

# Lazy-loaded singletons, one per detected language
_nlp_models: dict[str, Any] = {}


def _detect_language(text: str) -> str:
    """Return ISO 639-1 language code for *text*, defaulting to 'es' on failure."""
    try:
        from langdetect import detect, LangDetectException  # type: ignore[import]
        try:
            return detect(text)
        except LangDetectException:
            return "es"
    except ImportError:
        return "es"


def _get_nlp(lang: str = "es") -> Any:
    """Return (and lazily load) the SpaCy model for *lang*."""
    model_name = _MODELS.get(lang, "es_core_news_lg")
    if lang not in _nlp_models:
        logger.debug("[ner] Loading SpaCy model %r for lang=%r", model_name, lang)
        _nlp_models[lang] = spacy.load(model_name)
    return _nlp_models[lang]


def extract_entities(text: str) -> list[str]:
    """
    Clean the text and extract named entities + key noun chunks.
    Returns a deduplicated list of entity strings for RAG queries.
    """
    # Strip emojis, URLs and excessive punctuation
    clean = re.sub(r"http\S+", "", text)
    clean = re.sub(r"[^\w\s]", " ", clean)

    lang = _detect_language(clean)
    logger.debug("[ner] Detected language: %s", lang)
    doc = _get_nlp(lang)(clean)

    entities: list[str] = []

    # Named entities (PER, ORG, LOC, MISC)
    for ent in doc.ents:
        entities.append(ent.text.strip())

    # Key noun chunks as fallback when NER yields nothing
    if not entities:
        entities = [chunk.text.strip() for chunk in doc.noun_chunks]

    return list(dict.fromkeys(e for e in entities if len(e) > 2))  # deduplicate


def is_gibberish(text: str) -> bool:
    """
    Return True if *text* looks like random characters or incoherent input
    that would cause the RAG pipeline to hallucinate a verdict.

    Four signals are evaluated; at least TWO must fire simultaneously to
    avoid blocking legitimate short texts or partial English/Catalan input:

    1. alpha_ratio  < 0.40  — fewer than 40 % of tokens are real words
    2. avg_len      < 3.0   — average alpha-token length is very short
    3. few_meaningful       — fewer than 4 non-stop alpha tokens (len > 2)
    4. unknown_ratio > 0.60 — spaCy assigns POS=X to most tokens
    """
    clean = re.sub(r"http\S+", "", text)
    clean = re.sub(r"[^\w\s]", " ", clean).strip()

    if not clean:
        return True

    # ── Fast-path checks (no SpaCy needed) ────────────────────────────────────
    if len(clean) < 3:
        logger.debug("[ner] Gibberish fast-path triggered for text=%.20r", text)
        return True
    alnum_ratio = sum(c.isalnum() for c in clean) / len(clean)
    if alnum_ratio < 0.3:
        logger.debug("[ner] Gibberish fast-path triggered for text=%.20r", text)
        return True

    lang = _detect_language(clean)
    logger.debug("[ner] is_gibberish lang: %s", lang)
    doc = _get_nlp(lang)(clean)

    if not doc.has_annotation("TAG"):
        return False  # model could not parse at all — let pipeline handle it

    total = max(len(doc), 1)
    alpha_tokens = [t for t in doc if t.is_alpha]
    alpha_count = max(len(alpha_tokens), 1)

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
    return sum(conditions) >= 2
