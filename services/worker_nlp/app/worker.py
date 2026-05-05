"""
NLP Worker — consumes TextTask messages from RabbitMQ,
runs the RAG fact-checking pipeline, and delivers the verdict via Telegram.
"""
import asyncio
import os
import time

import aio_pika
from dotenv import load_dotenv

from shared.schemas import TextTask, NLPResult
from app.ner import extract_entities
from app.rag import hybrid_search, synthesize_verdict

load_dotenv()


async def process(message: aio_pika.abc.AbstractIncomingMessage) -> None:
    async with message.process():
        task = TextTask.model_validate_json(message.body)
        start = time.monotonic()

        entities = extract_entities(task.text)
        result: NLPResult = await hybrid_search(task.query_id, task.text, entities)
        result.summary = await synthesize_verdict(task.text, result)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # TODO: log to MongoDB + send Telegram reply
        print(f"[nlp] {task.query_id} → {result.verdict} ({elapsed_ms}ms)")


async def main() -> None:
    url = (
        f"amqp://{os.environ['RABBITMQ_USER']}:{os.environ['RABBITMQ_PASS']}"
        f"@{os.environ['RABBITMQ_HOST']}:{os.environ['RABBITMQ_PORT']}/"
    )
    connection = await aio_pika.connect_robust(url)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)
    queue = await channel.declare_queue(
        os.getenv("RABBITMQ_QUEUE_TEXTS", "topic_texts"), durable=True
    )
    await queue.consume(process)
    print("[nlp worker] Waiting for messages…")
    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
