"""
Named Entity Recognition using SpaCy es_core_news_lg.
Extracts key concepts from viral messages before RAG retrieval.
"""
from __future__ import annotations

import re
import spacy

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
