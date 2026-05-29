"""
Worker NLP - consume mensajes TextTask de RabbitMQ,
ejecutar el pipeline RAG de verificación de hechos y envía el veredicto por Telegram.
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
from app.rag import hybrid_search, synthesize_verdict, resolve_final_verdict, _is_no_info_response

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
    force=True,
)


def _sanitize_summary(summary: str) -> str:
    """Elimina el prefijo VEREDICTO: de la salida del LLM conservando la explicación.
    También limpia líneas 'Fuente:' / URLs inyectadas por rutas de fallback o alucinadas.
    """
    # 1. Elimina "VEREDICTO: FALSO — " / "VEREDICTO: VERDADERO — " / "VEREDICTO: NO VERIFICADO — "
    cleaned = re.sub(
        r"(?im)^[\s*]*veredicto?:\s*(?:falso|verdadero|no[\s_-]verificado)\s*[-–—]+\s*",
        "",
        summary,
    )
    # 2. Elimina una línea "VEREDICTO: XXXX" sin explicación.
    cleaned = re.sub(
        r"(?im)^[\s*]*veredicto?:\s*(?:falso|verdadero|no[\s_-]verificado)\s*$",
        "",
        cleaned,
    )
    # 3. Elimina keywords de veredicto al inicio del texto (eco del LLM sin prefijo).
    cleaned = re.sub(
        r"(?im)^(falso|verdadero|no[\s_-]verificado)\s*[-–—]+\s*",
        "",
        cleaned,
    )
    # 4. Elimina líneas 'Fuente:' espurias.
    cleaned = re.sub(r"(?im)^[\s*]*fuente:\s*.+$", "", cleaned)
    # 5. Elimina URLs sueltas alucinadas por el LLM.
    cleaned = re.sub(r"(?im)^https?://\S+\s*$", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _escape_html(text: str) -> str:
    """Escapa los tres caracteres reservados por el parser HTML de Telegram."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


async def _send_telegram_reply(chat_id: int, text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        logger.error("_send_telegram_reply: TELEGRAM_BOT_TOKEN no configurado - omitiendo respuesta a chat_id=%s", chat_id)
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
                "_send_telegram_reply: error de parseo HTML para chat_id=%s (%s) - reintentando como texto plano",
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
                    "_send_telegram_reply: fallback a texto plano también falló para chat_id=%s - %s: %s",
                    chat_id, type(inner_e).__name__, inner_e,
                )
        else:
            logger.error(
                "_send_telegram_reply: error enviando mensaje a chat_id=%s - %s: %s",
                chat_id, type(e).__name__, e,
            )
    except Exception as e:
        logger.error(
            "_send_telegram_reply: error enviando mensaje a chat_id=%s - %s: %s",
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
                logger.info("[nlp] Cache HIT para query_id=%s", task.query_id)
                elapsed_ms = int((time.monotonic() - start) * 1000)
            else:
                entities = extract_entities(task.text)
                result: NLPResult = await asyncio.wait_for(
                    hybrid_search(task.query_id, task.text, entities),
                    timeout=80.0,
                )
                result = await synthesize_verdict(result, task.text)

                _prev_verdict = result.verdict
                result = resolve_final_verdict(result)
                if result.verdict != _prev_verdict:
                    logger.info(
                        "process: veredicto resuelto %r → %r para query_id=%s",
                        _prev_verdict, result.verdict, task.query_id,
                    )

                elapsed_ms = int((time.monotonic() - start) * 1000)
                await set_cached_verdict(text_hash, result)

            nlp_processing_seconds.observe(elapsed_ms / 1000)

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
                    "[nlp worker] Error insertando en MongoDB para query_id=%s: %s",
                    task.query_id, mongo_err,
                )


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
            print(f"[nlp] Veredicto LLM:\n{result.summary}")

        except asyncio.TimeoutError:
            query_id = getattr(task, "query_id", "unknown")
            logger.error("[nlp worker] Timeout del pipeline RAG para query_id=%s", query_id)
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
                "[nlp worker] process() falló para query_id=%s: %s",
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
    logger.info("[nlp worker] Métricas Prometheus disponibles en :%d/metrics", metrics_port)
    logger.info("[nlp worker] Conectando a %s", mask_amqp_url(url))
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
            attempt = 0  # reiniciar en conexión exitosa
            print("[nlp worker] Esperando mensajes...")
            await asyncio.Future()
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                logger.error(
                    "[nlp worker] RabbitMQ inaccesible tras %d reintentos - abortando",
                    max_retries,
                )
                raise
            delay = min(2 ** attempt, 30)
            logger.error(
                "[nlp worker] Conexión RabbitMQ perdida: %s - reconectando en %ds (intento %d/%d)",
                e, delay, attempt, max_retries,
            )
            await asyncio.sleep(delay)


async def _run_smoke_test() -> None:
    """Envía un TextTask sintético a process() sin arrancar el servidor de métricas.
    Diseñado para invocarse con ``python -m app.worker --test`` dentro del contenedor.
    """
    import datetime
    import uuid

    class _FakeMsg:
        """Stub mínimo de mensaje aio_pika para las pruebas."""

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
        chat_id=0,  # 0 es falsy → process() omite la respuesta de Telegram
        text=(
            "El Gobierno de España ha confirmado que el nuevo virus detectado "
            "en Madrid no representa ningún peligro real para la población "
            "según fuentes oficiales del Ministerio de Sanidad."
        ),
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )
    logger.info("[nlp-test] Smoke test iniciado - query_id=%s", task.query_id)
    await process(_FakeMsg(task))
    logger.info("[nlp-test] Smoke test completado correctamente.")


if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser(description="Worker NLP")
    _parser.add_argument(
        "--test",
        action="store_true",
        help="Ejecuta un mensaje de smoke-test a través del pipeline y termina "
             "(omite el servidor Prometheus para evitar conflictos de puerto).",
    )
    _args = _parser.parse_args()

    if _args.test:
        asyncio.run(_run_smoke_test())
    else:
        asyncio.run(main())
