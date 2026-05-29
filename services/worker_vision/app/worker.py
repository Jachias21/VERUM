"""
Worker de visión - consume mensajes ImageTask de RabbitMQ,
ejecutar la inferencia forense y envía el veredicto al usuario por Telegram.
En caso de éxito hace ACK; ante excepción no controlada hace NACK sin
reencolar para que el mensaje no se pierda ni se reintente indefinidamente.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

import aio_pika
import cv2
import httpx
import numpy as np
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from telegram import Bot
from telegram.constants import ParseMode

from shared.schemas import ImageTask, QueryLog, VisionResult
from app.inference import run_inference
from app.xai import generate_heatmap

load_dotenv()

log = logging.getLogger(__name__)

# Cliente MongoDB singleton - un cliente por vida del proceso.
_mongo_client: AsyncIOMotorClient | None = None


def _get_db() -> AsyncIOMotorDatabase:
    global _mongo_client
    if _mongo_client is None:
        uri = (
            f"mongodb://{os.environ['MONGO_USER']}:{os.environ['MONGO_PASS']}"
            f"@{os.environ['MONGO_HOST']}:{os.environ['MONGO_PORT']}"
        )
        _mongo_client = AsyncIOMotorClient(uri)
    return _mongo_client[os.environ["MONGO_DB"]]


# Telegram image download

async def _download_telegram_image(file_id: str, token: str) -> bytes:
    """Descarga en dos pasos: GET /getFile para resolver file_path, luego descarga los bytes.
    Lanza httpx.HTTPError en errores de red.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        # Paso 1: resolver file_path
        r = await client.get(
            f"https://api.telegram.org/bot{token}/getFile",
            params={"file_id": file_id},
        )
        r.raise_for_status()
        file_path: str = r.json()["result"]["file_path"]

        # Paso 2: descargar contenido
        dl = await client.get(
            f"https://api.telegram.org/file/bot{token}/{file_path}"
        )
        dl.raise_for_status()
        return dl.content


def _build_reply_text(result: VisionResult) -> str:
    """Construye el texto de respuesta de Telegram en modo HTML."""
    score_pct = f"{result.ai_confidence_score:.1%}"
    prnu_line = (
        "🔬 PRNU signature: <b>detected ✓</b>"
        if result.prnu_detected
        else "🔬 PRNU signature: not detected"
    )

    if result.verdict == "FAKE":
        header = "⚠️ <b>FAKE image detected</b>"
    elif result.verdict == "REAL":
        header = "✅ <b>Image appears REAL</b>"
    else:
        header = "❓ <b>Verdict: UNVERIFIED</b>\n<i>The model is not yet available.</i>"

    return (
        f"{header}\n\n"
        f"🤖 AI confidence score: <code>{score_pct}</code>\n"
        f"{prnu_line}"
    )


def _image_resolution(image_bytes: bytes) -> str | None:
    """Devuelve la cadena 'AxA' o None si la imagen no puede decodificarse."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    return f"{w}x{h}"


async def _save_query_log(
    task: ImageTask,
    result: VisionResult,
    elapsed_ms: int,
    image_bytes: bytes,
) -> None:
    """Inserta un documento QueryLog. Registra y silencia cualquier error de BD."""
    try:
        db = _get_db()
        collection = db[os.environ["MONGO_COLLECTION_QUERIES"]]
        doc = QueryLog(
            query_id=str(task.query_id),
            timestamp=task.timestamp,
            user_hash=task.user_hash,
            payload_type="image",
            total_processing_time_ms=elapsed_ms,
            final_verdict=result.verdict,
            image_resolution=_image_resolution(image_bytes),
            ai_confidence_score=result.ai_confidence_score,
            prnu_detected=result.prnu_detected,
        )
        await collection.insert_one(doc.model_dump())
        log.debug("query_id=%s - QueryLog guardado en MongoDB.", task.query_id)
    except Exception as exc:  # noqa: BLE001
        log.error(
            "query_id=%s - error al guardar en MongoDB: %s",
            task.query_id, exc,
        )


async def _send_reply(
    bot: Bot,
    task: ImageTask,
    result: VisionResult,
    text: str,
) -> None:
    """
    Envía el veredicto al usuario:
      - FAKE  → send_photo con el heatmap y el texto como caption.
      - REAL / UNVERIFIED → send_message solo con texto.
    Si el envío de la foto falla (p.ej. archivo no disponible) cae al texto plano.
    """
    if result.verdict == "FAKE" and result.heatmap_path:
        try:
            with open(result.heatmap_path, "rb") as f:
                await bot.send_photo(
                    chat_id=task.chat_id,
                    photo=f,
                    caption=text,
                    parse_mode=ParseMode.HTML,
                )
            return
        except (OSError, Exception) as exc:  # noqa: BLE001
            log.warning(
                "query_id=%s - no se pudo enviar el heatmap (%s); "
                "enviando solo texto.",
                task.query_id, exc,
            )

    # Ruta no-FAKE o fallback
    await bot.send_message(
        chat_id=task.chat_id,
        text=text,
        parse_mode=ParseMode.HTML,
    )


async def process(message: aio_pika.abc.AbstractIncomingMessage) -> None:
    """
    Procesa un mensaje ImageTask de RabbitMQ.
    El context manager `message.process()` garantiza ACK en salida normal
    y NACK (requeue=False) ante excepción no controlada.
    """
    async with message.process():
        task = ImageTask.model_validate_json(message.body)
        start = time.monotonic()
        token = os.environ["TELEGRAM_BOT_TOKEN"]

        log.info(
            "query_id=%s - procesando file_id=%s para chat_id=%s",
            task.query_id, task.telegram_file_id, task.chat_id,
        )

        try:
            image_bytes = await _download_telegram_image(task.telegram_file_id, token)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "query_id=%s - descarga fallida (file_id=%s): %s",
                task.query_id, task.telegram_file_id, exc,
            )
            return

        result: VisionResult = await run_inference(task.query_id, image_bytes)

        if result.verdict == "FAKE":
            try:
                result.heatmap_path = await generate_heatmap(image_bytes, task.query_id)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "query_id=%s - error generando heatmap: %s",
                    task.query_id, exc,
                )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        text = _build_reply_text(result)
        async with Bot(token=token) as bot:
            await _send_reply(bot, task, result, text)

        await _save_query_log(task, result, elapsed_ms, image_bytes)

        log.info(
            "query_id=%s verdict=%s score=%.4f elapsed_ms=%d",
            task.query_id, result.verdict, result.ai_confidence_score, elapsed_ms,
        )


# Entry point

async def main() -> None:
    url = (
        f"amqp://{os.environ['RABBITMQ_USER']}:{os.environ['RABBITMQ_PASS']}"
        f"@{os.environ['RABBITMQ_HOST']}:{os.environ['RABBITMQ_PORT']}/"
    )
    connection = await aio_pika.connect_robust(url)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)
    queue = await channel.declare_queue(
        os.getenv("RABBITMQ_QUEUE_IMAGES", "topic_images"), durable=True
    )
    await queue.consume(process)
    log.info("[vision worker] Esperando mensajes en la cola '%s'...", queue.name)    await asyncio.Future()  # run forever


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    )
    asyncio.run(main())
