"""
Esquemas Pydantic compartidos para payloads de mensajes RabbitMQ y resultados entre servicios.
Todos los servicios importan desde este módulo para garantizar un contrato coherente.
"""
from __future__ import annotations

import uuid
import datetime
from typing import Literal

from pydantic import BaseModel, Field


# Tareas entrantes (Gateway -> Workers)

class ImageTask(BaseModel):
    query_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_hash: str                      # SHA-256 del user_id de Telegram (conforme RGPD)
    telegram_file_id: str
    chat_id: int
    timestamp: datetime.datetime


class TextTask(BaseModel):
    query_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_hash: str
    chat_id: int
    text: str
    timestamp: datetime.datetime


# Resultados de los workers

class VisionResult(BaseModel):
    query_id: uuid.UUID
    ai_confidence_score: float          # 0.0 = cámara real, 1.0 = sintético por IA
    prnu_detected: bool
    verdict: Literal["FAKE", "REAL", "UNVERIFIED"]
    heatmap_path: str | None = None     # Ruta local a la imagen de sobreposición Grad-CAM


class NLPResult(BaseModel):
    """Resultado del worker NLP.

    retrieved_context: texto bruto del artículo fuente recuperado de Qdrant/Google,
                       usado como entrada al LLM.
    summary:           veredicto final sintetizado por el LLM a partir de retrieved_context.
    """

    query_id: uuid.UUID
    extracted_entities: list[str]
    fact_check_matches: int
    source_url: str | None = None
    verdict: Literal["FAKE", "REAL", "UNVERIFIED"]
    retrieved_context: str = ""         # texto del artículo fuente enviado al LLM
    summary: str                        # veredicto de 3 líneas generado por el LLM


# Documento analítico de MongoDB

class QueryLog(BaseModel):
    query_id: str
    timestamp: datetime.datetime
    user_hash: str
    payload_type: Literal["image", "text"]
    total_processing_time_ms: int
    final_verdict: Literal["FAKE", "REAL", "UNVERIFIED"]
    cache_hit: bool = False
    # específico de imagen
    image_resolution: str | None = None
    ai_confidence_score: float | None = None
    prnu_detected: bool | None = None
    # específico de texto
    extracted_entities: list[str] | None = None
    fact_check_matches: int | None = None
    source_url: str | None = None
    feedback: str | None = None         # "correct" | "incorrect" - enviado vía comando /feedback
