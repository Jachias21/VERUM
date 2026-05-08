"""
NLP Worker — consumes TextTask messages from RabbitMQ,
runs the RAG fact-checking pipeline, and delivers the verdict via Telegram.
"""
import asyncio
import datetime
import logging
import os
import time

import aio_pika
from dotenv import load_dotenv
from telegram import Bot

logger = logging.getLogger(__name__)

from shared.db import get_mongo_db
from shared.schemas import TextTask, NLPResult, QueryLog
from app.ner import extract_entities
from app.rag import hybrid_search, synthesize_verdict

load_dotenv()


async def _send_telegram_reply(chat_id: int, text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        logger.error("_send_telegram_reply: TELEGRAM_BOT_TOKEN is not set — skipping reply to chat_id=%s", chat_id)
        return
    try:
        bot = Bot(token=token)
        await bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
    except Exception as e:
        logger.error(
            "_send_telegram_reply: failed to send message to chat_id=%s — %s: %s",
            chat_id, type(e).__name__, e,
        )


async def process(message: aio_pika.abc.AbstractIncomingMessage) -> None:
    async with message.process():
        task = TextTask.model_validate_json(message.body)
        start = time.monotonic()

        entities = extract_entities(task.text)
        result: NLPResult = await hybrid_search(task.query_id, task.text, entities)
        result.summary = await synthesize_verdict(task.text, result)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # ── GAP 1: Log to MongoDB ─────────────────────────────────────────────
        db = get_mongo_db()
        log = QueryLog(
            query_id=str(task.query_id),
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            user_hash=task.user_hash,
            payload_type="text",
            total_processing_time_ms=elapsed_ms,
            final_verdict=result.verdict,
            extracted_entities=result.extracted_entities,
            fact_check_matches=result.fact_check_matches,
            source_url=result.source_url,
        )
        await db["queries"].insert_one(log.model_dump())

        # ── GAP 2: Send Telegram reply ────────────────────────────────────────
        if task.chat_id:
            verdict_emoji = {"FAKE": "🔴", "REAL": "🟢", "UNVERIFIED": "🟡"}.get(result.verdict, "⚪")
            reply = (
                f"{verdict_emoji} *Veredicto: {result.verdict}*\n\n"
                f"{result.summary}\n\n"
                f"📎 Fuente: {result.source_url or 'Sin coincidencias en la base de datos.'}"
            )
            await _send_telegram_reply(task.chat_id, reply)

        print(f"[nlp] {task.query_id} → {result.verdict} ({elapsed_ms}ms)")
        print(f"[nlp] LLM verdict:\n{result.summary}")


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
