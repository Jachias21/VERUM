"""
Tests for shared/schemas.py — NLPResult field contract.
"""
from __future__ import annotations

import uuid
import pytest

from shared.schemas import NLPResult, QueryLog


def _make_nlp_result(**kwargs) -> NLPResult:
    defaults = dict(
        query_id=uuid.uuid4(),
        extracted_entities=[],
        fact_check_matches=0,
        verdict="UNVERIFIED",
        summary="",
    )
    defaults.update(kwargs)
    return NLPResult(**defaults)


# ── Test 1: retrieved_context field exists with default "" ───────────────────

def test_nlp_result_retrieved_context_default():
    result = _make_nlp_result()
    assert hasattr(result, "retrieved_context")
    assert result.retrieved_context == ""


# ── Test 2: retrieved_context and summary are independent ────────────────────

def test_nlp_result_fields_are_independent():
    result = _make_nlp_result(
        retrieved_context="Texto del artículo fuente.",
        summary="VEREDICTO: FALSO — esto es un bulo.",
    )
    assert result.retrieved_context == "Texto del artículo fuente."
    assert result.summary == "VEREDICTO: FALSO — esto es un bulo."

    # Mutate one; the other must not change
    result.summary = "Otro veredicto."
    assert result.retrieved_context == "Texto del artículo fuente."

    result.retrieved_context = "Nuevo contexto."
    assert result.summary == "Otro veredicto."


# ── Test 3: QueryLog does NOT have retrieved_context ────────────────────────

def test_query_log_has_no_retrieved_context():
    import datetime
    log = QueryLog(
        query_id=str(uuid.uuid4()),
        timestamp=datetime.datetime.now(datetime.timezone.utc),
        user_hash="a" * 64,
        payload_type="text",
        total_processing_time_ms=100,
        final_verdict="UNVERIFIED",
    )
    assert not hasattr(log, "retrieved_context"), (
        "QueryLog must not expose retrieved_context (privacy requirement)"
    )
