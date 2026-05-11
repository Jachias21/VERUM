"""
Named Entity Recognition using SpaCy es_core_news_lg.
Extracts key concepts from viral messages before RAG retrieval.
"""
from __future__ import annotations

import logging
import re
import spacy

logger = logging.getLogger("verum.ner")

_nlp = None  # lazy-loaded singleton


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("es_core_news_lg")
    return _nlp


def extract_entities(text: str) -> list[str]:
    """
    Clean the text and extract named entities + key noun chunks.
    Returns a deduplicated list of entity strings for RAG queries.
    """
    # Strip emojis, URLs and excessive punctuation
    clean = re.sub(r"http\S+", "", text)
    clean = re.sub(r"[^\w\s]", " ", clean)

    doc = _get_nlp()(clean)

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

    doc = _get_nlp()(clean)

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
