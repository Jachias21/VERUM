"""
Vision Worker — consumes ImageTask messages from RabbitMQ,
runs forensic inference, and delivers the verdict to the user via Telegram.
"""
import asyncio
import logging
import os
import time

import aio_pika
import cv2
import httpx
import numpy as np
from dotenv import load_dotenv
from telegram import Bot

from shared.db import get_mongo_db
from shared.schemas import ImageTask, QueryLog, VisionResult
from app.inference import run_inference
from app.xai import generate_heatmap

load_dotenv()

log = logging.getLogger(__name__)


async def _download_telegram_image(file_id: str, token: str) -> bytes:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            f"https://api.telegram.org/bot{token}/getFile",
            params={"file_id": file_id},
        )
        r.raise_for_status()
        file_path = r.json()["result"]["file_path"]
        dl = await client.get(
            f"https://api.telegram.org/file/bot{token}/{file_path}"
        )
        dl.raise_for_status()
        return dl.content


def _build_reply_text(result: VisionResult) -> str:
    score_pct = f"{result.ai_confidence_score:.1%}"
    prnu_line = "PRNU signature: " + ("detected ✓" if result.prnu_detected else "not detected")
    if result.verdict == "FAKE":
        header = "⚠️ *FAKE image detected*"
    elif result.verdict == "REAL":
        header = "✅ *Image appears REAL*"
    else:
        header = "❓ *Verdict: UNVERIFIED*"
    return f"{header}\n\nAI confidence score: `{score_pct}`\n{prnu_line}"


def _image_resolution(image_bytes: bytes) -> str | None:
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
    db = get_mongo_db()
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


async def process(message: aio_pika.abc.AbstractIncomingMessage) -> None:
    async with message.process():
        task = ImageTask.model_validate_json(message.body)
        start = time.monotonic()
        token = os.environ["TELEGRAM_BOT_TOKEN"]

        try:
            image_bytes = await _download_telegram_image(task.telegram_file_id, token)
        except Exception as exc:
            log.error(
                "query_id=%s — download failed (file_id=%s): %s",
                task.query_id, task.telegram_file_id, exc,
            )
            return  # message is ACK'd by the context manager; not requeued

        result: VisionResult = await run_inference(task.query_id, image_bytes)

        if result.verdict == "FAKE":
            result.heatmap_path = await generate_heatmap(image_bytes, task.query_id)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        text = _build_reply_text(result)
        async with Bot(token=token) as bot:
            if result.verdict == "FAKE" and result.heatmap_path:
                with open(result.heatmap_path, "rb") as f:
                    await bot.send_photo(
                        chat_id=task.chat_id,
                        photo=f,
                        caption=text,
                        parse_mode="Markdown",
                    )
            else:
                await bot.send_message(
                    chat_id=task.chat_id,
                    text=text,
                    parse_mode="Markdown",
                )

        await _save_query_log(task, result, elapsed_ms, image_bytes)

        log.info(
            "query_id=%s verdict=%s elapsed_ms=%d",
            task.query_id, result.verdict, elapsed_ms,
        )


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
    log.info("[vision worker] Waiting for messages…")
    await asyncio.Future()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
