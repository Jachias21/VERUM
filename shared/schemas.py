"""
Shared Pydantic schemas for RabbitMQ message payloads and inter-service results.
All services import from this module to guarantee a consistent contract.
"""
from __future__ import annotations

import uuid
import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ── Inbound tasks (Gateway → Workers) ────────────────────────────────────────

class ImageTask(BaseModel):
    query_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_hash: str                      # SHA-256 of Telegram user_id (GDPR-safe)
    telegram_file_id: str
    timestamp: datetime.datetime


class TextTask(BaseModel):
    query_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_hash: str
    chat_id: int
    text: str
    timestamp: datetime.datetime


# ── Worker results ────────────────────────────────────────────────────────────

class VisionResult(BaseModel):
    query_id: uuid.UUID
    ai_confidence_score: float          # 0.0 = real camera, 1.0 = synthetic AI
    prnu_detected: bool
    verdict: Literal["FAKE", "REAL", "UNVERIFIED"]
    heatmap_path: str | None = None     # Local path to Grad-CAM overlay image


class NLPResult(BaseModel):
    """NLP worker result.

    retrieved_context: raw text of the source article retrieved from Qdrant/Google,
                       used as input to the LLM.
    summary:           final verdict synthesised by the LLM from retrieved_context.
    """

    query_id: uuid.UUID
    extracted_entities: list[str]
    fact_check_matches: int
    source_url: str | None = None
    verdict: Literal["FAKE", "REAL", "UNVERIFIED"]
    retrieved_context: str = ""         # source article text fed to the LLM
    summary: str                        # LLM-generated 3-line verdict


# ── MongoDB analytics document ────────────────────────────────────────────────

class QueryLog(BaseModel):
    query_id: str
    timestamp: datetime.datetime
    user_hash: str
    payload_type: Literal["image", "text"]
    total_processing_time_ms: int
    final_verdict: Literal["FAKE", "REAL", "UNVERIFIED"]
    # image-specific
    image_resolution: str | None = None
    ai_confidence_score: float | None = None
    prnu_detected: bool | None = None
    # text-specific
    extracted_entities: list[str] | None = None
    fact_check_matches: int | None = None
    source_url: str | None = None
