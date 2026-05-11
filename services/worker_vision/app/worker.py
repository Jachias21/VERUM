"""
Vision Worker — consumes ImageTask messages from RabbitMQ,
runs forensic inference, and delivers the verdict to the user via Telegram.

Message flow:
  1. Consume ImageTask from RabbitMQ queue (topic_images).
  2. Download image from Telegram (getFile → file download).
  3. Run dual-branch ONNX inference (run_inference).
  4. If FAKE → generate Grad-CAM heatmap (generate_heatmap).
  5. Reply to user via Telegram (photo + caption if FAKE, text otherwise).
  6. Persist QueryLog to MongoDB.
  7. ACK the RabbitMQ message (always, even on download/send failure).
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

# ---------------------------------------------------------------------------
# MongoDB singleton — one client for the lifetime of the process.
# get_mongo_db() from shared/db.py creates a new client on every call;
# we keep our own handle here to avoid per-message connection overhead.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Telegram image download
# ---------------------------------------------------------------------------

async def _download_telegram_image(file_id: str, token: str) -> bytes:
    """
    Two-step Telegram download:
      1. GET /getFile?file_id=... → obtain file_path.
      2. GET /file/bot{token}/{file_path} → download raw bytes.

    Raises httpx.HTTPError on any HTTP or network error.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        # Step 1 — resolve file_path
        r = await client.get(
            f"https://api.telegram.org/bot{token}/getFile",
            params={"file_id": file_id},
        )
        r.raise_for_status()
        file_path: str = r.json()["result"]["file_path"]

        # Step 2 — download content
        dl = await client.get(
            f"https://api.telegram.org/file/bot{token}/{file_path}"
        )
        dl.raise_for_status()
        return dl.content


# ---------------------------------------------------------------------------
# Reply text builder
# ---------------------------------------------------------------------------

def _build_reply_text(result: VisionResult) -> str:
    """
    Builds the Telegram reply text using HTML parse mode to avoid
    MarkdownV2 escaping issues with score percentages and symbols.
    """
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_resolution(image_bytes: bytes) -> str | None:
    """Returns 'WxH' string or None if the image cannot be decoded."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    return f"{w}x{h}"


# ---------------------------------------------------------------------------
# MongoDB persistence
# ---------------------------------------------------------------------------

async def _save_query_log(
    task: ImageTask,
    result: VisionResult,
    elapsed_ms: int,
    image_bytes: bytes,
) -> None:
    """Inserts a QueryLog document. Logs and swallows any DB error."""
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
        log.debug("query_id=%s — QueryLog guardado en MongoDB.", task.query_id)
    except Exception as exc:  # noqa: BLE001
        log.error(
            "query_id=%s — error al guardar en MongoDB: %s",
            task.query_id, exc,
        )


# ---------------------------------------------------------------------------
# Telegram reply
# ---------------------------------------------------------------------------

async def _send_reply(
    bot: Bot,
    task: ImageTask,
    result: VisionResult,
    text: str,
) -> None:
    """
    Sends the verdict to the user:
      - FAKE  → send_photo with heatmap as attachment and text as caption.
      - REAL / UNVERIFIED → send_message with text only.

    If sending the photo fails (e.g. heatmap file missing) it falls back
    to sending a plain text message.
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
                "query_id=%s — no se pudo enviar el heatmap (%s); "
                "enviando solo texto.",
                task.query_id, exc,
            )

    # Fallback / non-FAKE path
    await bot.send_message(
        chat_id=task.chat_id,
        text=text,
        parse_mode=ParseMode.HTML,
    )


# ---------------------------------------------------------------------------
# RabbitMQ message handler
# ---------------------------------------------------------------------------

async def process(message: aio_pika.abc.AbstractIncomingMessage) -> None:
    """
    Processes one RabbitMQ ImageTask message.

    The `message.process()` context manager guarantees ACK on normal exit
    and NACK (with requeue=False) on unhandled exception, so the message
    is never lost but also never requeued for infinite retries.
    """
    async with message.process():
        task = ImageTask.model_validate_json(message.body)
        start = time.monotonic()
        token = os.environ["TELEGRAM_BOT_TOKEN"]

        log.info(
            "query_id=%s — procesando file_id=%s para chat_id=%s",
            task.query_id, task.telegram_file_id, task.chat_id,
        )

        # ── Step 1: Download image ────────────────────────────────────────────
        try:
            image_bytes = await _download_telegram_image(task.telegram_file_id, token)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "query_id=%s — descarga fallida (file_id=%s): %s",
                task.query_id, task.telegram_file_id, exc,
            )
            # ACK implícito: message.process() ya lo gestiona al salir limpiamente.
            return

        # ── Step 2: Inference ─────────────────────────────────────────────────
        result: VisionResult = await run_inference(task.query_id, image_bytes)

        # ── Step 3: Grad-CAM heatmap (solo si es FAKE) ───────────────────────
        if result.verdict == "FAKE":
            try:
                result.heatmap_path = await generate_heatmap(image_bytes, task.query_id)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "query_id=%s — error generando heatmap: %s",
                    task.query_id, exc,
                )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # ── Step 4: Reply to user ─────────────────────────────────────────────
        text = _build_reply_text(result)
        async with Bot(token=token) as bot:
            await _send_reply(bot, task, result, text)

        # ── Step 5: Persist to MongoDB ────────────────────────────────────────
        await _save_query_log(task, result, elapsed_ms, image_bytes)

        log.info(
            "query_id=%s verdict=%s score=%.4f elapsed_ms=%d",
            task.query_id, result.verdict, result.ai_confidence_score, elapsed_ms,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
    log.info("[vision worker] Esperando mensajes en la cola '%s'…", queue.name)
    await asyncio.Future()  # run forever


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    asyncio.run(main())
