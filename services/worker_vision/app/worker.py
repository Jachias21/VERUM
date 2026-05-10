"""
Vision Worker — consumes ImageTask messages from RabbitMQ,
runs forensic inference, and delivers the verdict to the user via Telegram.
"""
import asyncio
import datetime
import logging
import os
import time

import aio_pika
import cv2
import numpy as np
from dotenv import load_dotenv

from shared.db import get_mongo_db
from shared.schemas import ImageTask, QueryLog, VisionResult
from app.inference import run_inference
from app.xai import generate_heatmap

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
    force=True,
)

logger = logging.getLogger("verum.vision")


def _get_resolution(image_bytes: bytes) -> str | None:
    """Decode image bytes and return resolution as 'WxH', or None on failure."""
    if not image_bytes:
        return None
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        return f"{w}x{h}"
    except Exception:
        return None


async def process(message: aio_pika.abc.AbstractIncomingMessage) -> None:
    async with message.process():
        task = ImageTask.model_validate_json(message.body)
        start = time.monotonic()

        # TODO: download image from Telegram using task.telegram_file_id
        image_bytes: bytes = b""  # placeholder

        result: VisionResult = await run_inference(task.query_id, image_bytes)

        if result.verdict == "FAKE":
            result.heatmap_path = await generate_heatmap(image_bytes, task.query_id)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        logger.info(
            "[vision] %s → %s | score=%.3f | prnu=%s | %dms",
            task.query_id, result.verdict, result.ai_confidence_score,
            result.prnu_detected, elapsed_ms,
        )
        print(f"[vision] {task.query_id} → {result.verdict} ({elapsed_ms}ms)")

        try:
            db = get_mongo_db()
            log = QueryLog(
                query_id=str(task.query_id),
                timestamp=datetime.datetime.now(datetime.timezone.utc),
                user_hash=task.user_hash,
                payload_type="image",
                total_processing_time_ms=elapsed_ms,
                final_verdict=result.verdict,
                image_resolution=_get_resolution(image_bytes),
                ai_confidence_score=result.ai_confidence_score,
                prnu_detected=result.prnu_detected,
            )
            await db["queries"].insert_one(log.model_dump())
        except Exception as e:
            logger.error("[vision] MongoDB insert failed for query_id=%s: %s", task.query_id, e)


async def main() -> None:
    url = (
        f"amqp://{os.environ['RABBITMQ_USER']}:{os.environ['RABBITMQ_PASS']}"
        f"@{os.environ['RABBITMQ_HOST']}:{os.environ['RABBITMQ_PORT']}/"
    )
    connection = await aio_pika.connect_robust(url)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)   # process one image at a time
    queue = await channel.declare_queue(
        os.getenv("RABBITMQ_QUEUE_IMAGES", "topic_images"), durable=True
    )
    await queue.consume(process)
    print("[vision worker] Waiting for messages…")
    await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
