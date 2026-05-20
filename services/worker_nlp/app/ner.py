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

    # ── Named entities (always included) ─────────────────────────────────────
    ner_entities: list[str] = [ent.text.strip() for ent in doc.ents]
    ner_lower: set[str] = {e.lower() for e in ner_entities}

    # ── Noun chunks (always combined, not just as fallback) ───────────────────
    filtered_chunks: list[str] = []
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        # Must be at least 3 chars and have a non-stopword head
        if len(chunk_text) < 3 or chunk.root.is_stop:
            continue
        chunk_lower = chunk_text.lower()
        # Skip if already covered by (or covering) a named entity
        if any(chunk_lower in ne or ne in chunk_lower for ne in ner_lower):
            continue
        filtered_chunks.append(chunk_text)

    # Sort chunks by length descending so longer (more informative) ones fill slots first
    filtered_chunks.sort(key=len, reverse=True)

    # Combine: NER first, then chunks, capped at 8
    MAX_ENTITIES = 8
    combined = ner_entities + filtered_chunks
    result = list(dict.fromkeys(e for e in combined if len(e) > 2))[:MAX_ENTITIES]

    logger.info("[ner] Extracted %d entities: %s", len(result), result)
    return result


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

    # ── Vowel fast-path: most alphabetic tokens lack any vowel → gibberish ────
    _VOWELS = set("aeiouáéíóúü")
    ws_alpha = [t for t in clean.split() if t.isalpha()]
    if ws_alpha:
        no_vowel_count = sum(1 for t in ws_alpha if not _VOWELS.intersection(t.lower()))
        # With >= 5 tokens we have enough statistical signal to lower the threshold;
        # keyboard-mash patterns (e.g. "asdf qwer zxcv") have ~40 % vowel-less tokens.
        threshold = 0.40 if len(ws_alpha) >= 5 else 0.50
        if no_vowel_count / len(ws_alpha) >= threshold:
            logger.debug("[ner] Gibberish vowel fast-path triggered for text=%.20r", text)
            return True

    lang = _detect_language(clean)
    logger.debug("[ner] is_gibberish lang: %s", lang)
    doc = _get_nlp(lang)(clean)

    if not doc.has_annotation("TAG"):
        return False  # model could not parse at all — let pipeline handle it

    total = max(len(doc), 1)
    alpha_tokens = [t for t in doc if t.is_alpha]
    alpha_count = max(len(alpha_tokens), 1)

    # ── New signal 1: Out-of-Vocabulary (OOV) ratio ───────────────────────────
    # Only meaningful when the model has word vectors (es_core_news_lg does).
    if doc.vocab.vectors_length > 0:
        long_alpha = [t for t in alpha_tokens if len(t.text) > 2]
        if len(long_alpha) >= 4:
            oov_count = sum(1 for t in long_alpha if t.is_oov)
            oov_ratio = oov_count / len(long_alpha)
            if oov_ratio >= 0.70:
                logger.debug("[ner] Gibberish OOV fast-path: oov_ratio=%.2f", oov_ratio)
                return True

    # ── New signal 2: Consecutive-consonant ratio (pseudo-acronym / random mash)
    # ñ is treated as a consonant for Spanish; pattern is language-agnostic.
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
