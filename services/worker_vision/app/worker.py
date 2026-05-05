"""
Vision Worker — consumes ImageTask messages from RabbitMQ,
runs forensic inference, and delivers the verdict to the user via Telegram.
"""
import asyncio
import os
import time

import aio_pika
from dotenv import load_dotenv

from shared.schemas import ImageTask, VisionResult
from app.inference import run_inference
from app.xai import generate_heatmap

load_dotenv()


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

        # TODO: log to MongoDB + send Telegram reply
        print(f"[vision] {task.query_id} → {result.verdict} ({elapsed_ms}ms)")


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
