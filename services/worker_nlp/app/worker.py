"""
NLP Worker — consumes TextTask messages from RabbitMQ,
runs the RAG fact-checking pipeline, and delivers the verdict via Telegram.
"""
import asyncio
import datetime
import hashlib
import logging
import os
import re
import time

import aio_pika
from dotenv import load_dotenv
from prometheus_client import start_http_server
import telegram.error
from telegram import Bot

logger = logging.getLogger(__name__)

from shared.db import get_mongo_db
from shared.rabbitmq_utils import build_amqp_url, mask_amqp_url
from shared.schemas import TextTask, NLPResult, QueryLog
from app.cache import get_cached_verdict, set_cached_verdict
from app.metrics import nlp_processing_seconds
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


def _sanitize_summary(summary: str) -> str:
    """Strip the VEREDICTO: prefix from LLM output while preserving the explanation.

    The LLM often responds in a single line:
        "VEREDICTO: FALSO — Según el artículo, Mercadona desmintió..."
    or without the prefix:
        "FALSO — Según el artículo..."

    We strip only the verdict keyword and keep the explanation that follows the
    separator (' — ', ' - ', ' – ').  If there is no separator we strip the
    whole VEREDICTO: line only when it stands alone (no explanation text).
    We also strip spurious 'Fuente:' / URL lines injected by fallback paths or
    hallucinated by the LLM.
    """
    # 1. Strip "VEREDICTO: FALSO — " / "VEREDICTO: VERDADERO — " / "VEREDICTO: NO VERIFICADO — "
    #    keeping everything after the separator.
    cleaned = re.sub(
        r"(?im)^[\s*]*veredicto?:\s*(?:falso|verdadero|no[\s_-]verificado)\s*[-–—]+\s*",
        "",
        summary,
    )
    # 2. Strip a bare "VEREDICTO: XXXX" line that has NO explanation after it.
    cleaned = re.sub(
        r"(?im)^[\s*]*veredicto?:\s*(?:falso|verdadero|no[\s_-]verificado)\s*$",
        "",
        cleaned,
    )
    # 3. Strip bare verdict keywords at the very start of the text (LLM echo without prefix).
    cleaned = re.sub(
        r"(?im)^(falso|verdadero|no[\s_-]verificado)\s*[-–—]+\s*",
        "",
        cleaned,
    )
    # 4. Strip 'Fuente:' lines (spurious from fallback paths or hallucinated by LLM).
    cleaned = re.sub(r"(?im)^[\s*]*fuente:\s*.+$", "", cleaned)
    # 5. Strip bare URLs the LLM may hallucinate.
    cleaned = re.sub(r"(?im)^https?://\S+\s*$", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _escape_html(text: str) -> str:
    """Escape the three characters reserved by Telegram's HTML parser."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


async def _send_telegram_reply(chat_id: int, text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        logger.error("_send_telegram_reply: TELEGRAM_BOT_TOKEN is not set — skipping reply to chat_id=%s", chat_id)
        return
    bot = Bot(token=token)
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
    except telegram.error.BadRequest as e:
        err_str = str(e).lower()
        if "parse" in err_str or "entity" in err_str or "can't find end" in err_str:
            logger.warning(
                "_send_telegram_reply: HTML parse error for chat_id=%s (%s) — retrying as plain text",
                chat_id, e,
            )
            try:
                plain = re.sub(r"<[^>]+>", "", text)
                await bot.send_message(
                    chat_id=chat_id,
                    text=plain,
                    disable_web_page_preview=True,
                )
            except Exception as inner_e:
                logger.error(
                    "_send_telegram_reply: plain-text fallback also failed for chat_id=%s — %s: %s",
                    chat_id, type(inner_e).__name__, inner_e,
                )
        else:
            logger.error(
                "_send_telegram_reply: failed to send message to chat_id=%s — %s: %s",
                chat_id, type(e).__name__, e,
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

            text_hash = hashlib.sha256(task.text.strip().lower().encode()).hexdigest()
            cached = await get_cached_verdict(text_hash)
            cache_hit = cached is not None
            if cache_hit:
                result = cached
                logger.info("[nlp] Cache HIT for query_id=%s", task.query_id)
                elapsed_ms = int((time.monotonic() - start) * 1000)
            else:
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
                await set_cached_verdict(text_hash, result)

            # ── Prometheus: observe end-to-end NLP latency ─────────────────────
            nlp_processing_seconds.observe(elapsed_ms / 1000)

            # ── GAP 1: Log to MongoDB ─────────────────────────────────────────────
            db = get_mongo_db()
            log = QueryLog(
                query_id=str(task.query_id),
                timestamp=datetime.datetime.now(datetime.timezone.utc),
                user_hash=task.user_hash,
                payload_type="text",
                total_processing_time_ms=elapsed_ms,
                final_verdict=result.verdict,
                cache_hit=cache_hit,
                extracted_entities=result.extracted_entities,
                fact_check_matches=result.fact_check_matches,
                source_url=result.source_url,
            )
            try:
                await db["queries"].insert_one(log.model_dump())
            except Exception as mongo_err:
                logger.error(
                    "[nlp worker] MongoDB insert failed for query_id=%s: %s",
                    task.query_id, mongo_err,
                )

            # ── GAP 2: Send Telegram reply ────────────────────────────────────────
            if task.chat_id:
                _no_info_override = not cache_hit and _is_no_info_response(result.summary)
                if _no_info_override:
                    reply = "🟡 <b>No encontré información verificada sobre esto.</b>\n\n" + _escape_html(result.summary)
                else:
                    verdict_emoji = {"FAKE": "🔴", "REAL": "🟢", "UNVERIFIED": "🟡"}.get(result.verdict, "⚪")
                    _clean_summary = _escape_html(_sanitize_summary(result.summary))
                    if result.verdict == "UNVERIFIED" and not result.source_url:
                        _source_line = (
                            "💡 No encontramos esta afirmación en fuentes verificadas.\n"
                            "Recomendamos comprobarlo manualmente en:\n"
                            "• maldita.es · newtral.es · snopes.com · factcheck.org"
                        )
                    else:
                        _source_line = "📎 Fuente: " + _escape_html(
                            result.source_url or "Sin coincidencias en la base de datos."
                        )
                    reply = (
                        f"{verdict_emoji} <b>Veredicto: {result.verdict}</b>\n\n"
                        f"{_clean_summary}\n\n"
                        f"{_source_line}"
                    )
                    if cache_hit:
                        reply += "\n\n⚡ <i>(respuesta cacheada)</i>"
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
    url = build_amqp_url()
    max_retries = int(os.getenv("RABBITMQ_CONNECT_MAX_RETRIES", "10"))
    metrics_port = int(os.getenv("NLP_METRICS_PORT", "9101"))
    start_http_server(metrics_port)
    logger.info("[nlp worker] Prometheus metrics available on :%d/metrics", metrics_port)
    logger.info("[nlp worker] Connecting to %s", mask_amqp_url(url))
    attempt = 0
    while True:
        try:
            connection = await aio_pika.connect_robust(url)
            channel = await connection.channel()
            await channel.set_qos(prefetch_count=1)
            queue = await channel.declare_queue(
                os.getenv("RABBITMQ_QUEUE_TEXTS", "topic_texts"), durable=True
            )
            await queue.consume(process)
            attempt = 0  # reset on successful connection
            print("[nlp worker] Waiting for messages…")
            await asyncio.Future()
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                logger.error(
                    "[nlp worker] RabbitMQ unreachable after %d retries — giving up",
                    max_retries,
                )
                raise
            delay = min(2 ** attempt, 30)
            logger.error(
                "[nlp worker] RabbitMQ connection lost: %s — reconnecting in %ds (attempt %d/%d)",
                e, delay, attempt, max_retries,
            )
            await asyncio.sleep(delay)


async def _run_smoke_test() -> None:
    """Push one synthetic TextTask through process() without starting the metrics server.

    Designed to be invoked via ``python -m app.worker --test`` inside a running
    container so that ``make nlp-test`` can validate the pipeline without
    conflicting with the already-bound Prometheus port.
    """
    import datetime
    import uuid

    class _FakeMsg:
        """Minimal aio_pika message stub — mirrors FakeMessage in the test suite."""

        def __init__(self, task: TextTask) -> None:
            self.body = task.model_dump_json().encode()

        def process(self):  # noqa: ANN201
            class _Ctx:
                async def __aenter__(self_) -> None:  # noqa: N805
                    return None

                async def __aexit__(self_, *_) -> None:  # noqa: N805
                    return None

            return _Ctx()

    task = TextTask(
        query_id=uuid.uuid4(),
        user_hash="a" * 64,
        chat_id=0,  # 0 is falsy → process() skips Telegram reply
        text=(
            "El Gobierno de España ha confirmado que el nuevo virus detectado "
            "en Madrid no representa ningún peligro real para la población "
            "según fuentes oficiales del Ministerio de Sanidad."
        ),
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )
    logger.info("[nlp-test] Smoke test started — query_id=%s", task.query_id)
    await process(_FakeMsg(task))
    logger.info("[nlp-test] Smoke test completed successfully.")


if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser(description="NLP Worker")
    _parser.add_argument(
        "--test",
        action="store_true",
        help="Run a single smoke-test message through the pipeline and exit "
             "(skips Prometheus server startup to avoid port conflicts).",
    )
    _args = _parser.parse_args()

    if _args.test:
        asyncio.run(_run_smoke_test())
    else:
        asyncio.run(main())
