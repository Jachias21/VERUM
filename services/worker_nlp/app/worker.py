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
from app.ner import extract_entities, is_gibberish
from app.rag import hybrid_search, synthesize_verdict, _extract_verdict_from_llm_output

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
    force=True,
)


_NO_INFO_PHRASES = (
    "no tengo información",
    "no menciona",
    "no puedo confirmar",
    "no hay información",
    "no se menciona",
    "sin información",
    # English equivalents
    "no information",
    "cannot confirm",
    "can't confirm",
    "no relevant",
)


def _is_no_info_response(text: str) -> bool:
    """Return True if the LLM summary indicates it could not find relevant info."""
    lower = text.lower()
    return any(phrase in lower for phrase in _NO_INFO_PHRASES)


async def _send_telegram_reply(chat_id: int, text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        logger.error("_send_telegram_reply: TELEGRAM_BOT_TOKEN is not set — skipping reply to chat_id=%s", chat_id)
        return
    try:
        bot = Bot(token=token)
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )
    except Exception as e:
        logger.error(
            "_send_telegram_reply: failed to send message to chat_id=%s — %s: %s",
            chat_id, type(e).__name__, e,
        )


async def process(message: aio_pika.abc.AbstractIncomingMessage) -> None:
    async with message.process():
        task: TextTask | None = None
        try:
            task = TextTask.model_validate_json(message.body)
            start = time.monotonic()

            if is_gibberish(task.text):
                logger.warning(
                    "[nlp] Gibberish detected for query_id=%s, skipping pipeline",
                    task.query_id,
                )
                if task.chat_id:
                    await _send_telegram_reply(
                        task.chat_id,
                        "⚠️ No he podido entender tu mensaje. Por favor, envíame una noticia o afirmación completa en español o inglés.",
                    )
                return

            entities = extract_entities(task.text)
            result: NLPResult = await asyncio.wait_for(
                hybrid_search(task.query_id, task.text, entities),
                timeout=80.0,
            )
            result.summary = await synthesize_verdict(task.text, result)

            # Override reply when the LLM had no relevant context to work with
            _no_info_override = _is_no_info_response(result.summary)
            if _no_info_override:
                result.verdict = "UNVERIFIED"
                result.summary = (
                    "No encontré información verificada sobre esto.\n\n"
                    "No tenemos esta noticia en nuestra base de datos ni en fuentes de "
                    "fact-checking. Esto no significa que sea falsa — simplemente no hay registros.\n\n"
                    "💡 Comprueba en: maldita.es · newtral.es · snopes.com"
                )

            # Override verdict with LLM output when it follows the expected format
            llm_verdict = _extract_verdict_from_llm_output(result.summary)
            if llm_verdict is not None:
                logger.info(
                    "process: LLM overriding RAG verdict %r → %r for query_id=%s",
                    result.verdict, llm_verdict, task.query_id,
                )
                result.verdict = llm_verdict

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
                if _no_info_override:
                    reply = "🟡 *No encontré información verificada sobre esto.*\n\n" + result.summary
                else:
                    verdict_emoji = {"FAKE": "🔴", "REAL": "🟢", "UNVERIFIED": "🟡"}.get(result.verdict, "⚪")
                    reply = (
                        f"{verdict_emoji} *Veredicto: {result.verdict}*\n\n"
                        f"{result.summary}\n\n"
                        f"📎 Fuente: {result.source_url or 'Sin coincidencias en la base de datos.'}"
                    )
                await _send_telegram_reply(task.chat_id, reply)

            print(f"[nlp] {task.query_id} → {result.verdict} ({elapsed_ms}ms)")
            print(f"[nlp] LLM verdict:\n{result.summary}")

        except asyncio.TimeoutError:
            query_id = getattr(task, "query_id", "unknown")
            logger.error("[nlp worker] RAG pipeline timeout for query_id=%s", query_id)
            if task is not None and task.chat_id:
                try:
                    await _send_telegram_reply(
                        task.chat_id,
                        "⚠️ Ha ocurrido un error al procesar tu mensaje. Por favor, inténtalo de nuevo en unos momentos.",
                    )
                except Exception:
                    pass

        except Exception as e:
            query_id = getattr(task, "query_id", "unknown")
            logger.error(
                "[nlp worker] process() failed for query_id=%s: %s",
                query_id, e, exc_info=True,
            )
            if task is not None and task.chat_id:
                try:
                    await _send_telegram_reply(
                        task.chat_id,
                        "⚠️ Ha ocurrido un error al procesar tu mensaje. Por favor, inténtalo de nuevo en unos momentos.",
                    )
                except Exception:
                    pass


async def main() -> None:
    url = (
        f"amqp://{os.environ['RABBITMQ_USER']}:{os.environ['RABBITMQ_PASS']}"
        f"@{os.environ['RABBITMQ_HOST']}:{os.environ['RABBITMQ_PORT']}/"
    )
    while True:
        try:
            connection = await aio_pika.connect_robust(url)
            channel = await connection.channel()
            await channel.set_qos(prefetch_count=1)
            queue = await channel.declare_queue(
                os.getenv("RABBITMQ_QUEUE_TEXTS", "topic_texts"), durable=True
            )
            await queue.consume(process)
            print("[nlp worker] Waiting for messages…")
            await asyncio.Future()
        except Exception as e:
            logger.error("[nlp worker] RabbitMQ connection lost: %s — reconnecting in 5s", e)
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
