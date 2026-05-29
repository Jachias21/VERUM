"""
Enrutador de mensajes - clasifica los payloads entrantes de Telegram y publica
tareas en la cola RabbitMQ correspondiente.
"""
import asyncio
import os
import hashlib
import logging
import uuid
from datetime import datetime, timezone

import aio_pika
from telegram import Bot

from app.metrics import texts_received, images_received, messages_rejected
from app.rate_limiter import check_rate_limit
from shared.rabbitmq_utils import build_amqp_url, mask_amqp_url

logger = logging.getLogger(__name__)


async def _send_interim_message(chat_id: int, text: str) -> None:
    """Envía un acuse de recibo al usuario vía Telegram (fire-and-forget)."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        logger.warning(
            "_send_interim_message: TELEGRAM_BOT_TOKEN no configurado - omitiendo interim a chat_id=%s",
            chat_id,
        )
        return
    try:
        bot = Bot(token=token)
        await bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "_send_interim_message: error al enviar interim a chat_id=%s - %s: %s",
            chat_id, type(e).__name__, e,
        )


_connection: aio_pika.abc.AbstractRobustConnection | None = None
_amqp_url_logged = False


async def _get_channel() -> aio_pika.abc.AbstractChannel:
    global _connection, _amqp_url_logged
    url = build_amqp_url()
    if not _amqp_url_logged:
        logger.info("gateway: AMQP target = %s", mask_amqp_url(url))
        _amqp_url_logged = True
    if _connection is None or _connection.is_closed:
        _connection = await aio_pika.connect_robust(url)
    channel = await _connection.channel()
    # Declarar las colas como durable para que los mensajes sobrevivan reinicios del broker
    # y no se pierdan si el gateway arranca antes que los workers.
    await channel.declare_queue(os.getenv("RABBITMQ_QUEUE_IMAGES", "topic_images"), durable=True)
    await channel.declare_queue(os.getenv("RABBITMQ_QUEUE_TEXTS", "topic_texts"), durable=True)
    return channel


async def route_message(payload: dict) -> None:
    message = payload.get("message") or payload.get("channel_post", {})
    if not message:
        logger.warning("route_message: el payload no contiene clave 'message' ni 'channel_post' - omitiendo. payload=%s", payload)
        messages_rejected.inc()
        return

    user_id = str(message.get("from", {}).get("id", "unknown"))
    user_hash = hashlib.sha256(user_id.encode()).hexdigest()
    chat_id: int = message.get("chat", {}).get("id", 0)
    query_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    if "photo" in message:
        from shared.schemas import ImageTask
        file_id = message["photo"][-1]["file_id"]   # resolución máxima
        task = ImageTask(
            query_id=query_id,
            user_hash=user_hash,
            telegram_file_id=file_id,
            chat_id=chat_id,
            timestamp=timestamp,
        )
        queue = os.getenv("RABBITMQ_QUEUE_IMAGES", "topic_images")
        body = task.model_dump_json()

    elif "text" in message:
        from shared.schemas import TextTask
        text: str = message.get("text", "")
        chat_id: int = message.get("chat", {}).get("id", 0)
        min_length = int(os.getenv("NLP_MIN_TEXT_LENGTH", 50))
        if len(text) < min_length:
            logger.info(
                "route_message: texto demasiado corto (%d chars, mín=%d) - descartado. chat_id=%s",
                len(text), min_length, chat_id,
            )
            messages_rejected.inc()
            if chat_id:
                await _send_interim_message(
                    chat_id,
                    "⚠️ Tu mensaje es muy corto para analizarlo. Envíame un texto de al menos "
                    f"{min_length} caracteres con una afirmación o noticia completa.",
                )
            return
        if not text.startswith("/"):
            asyncio.create_task(
                _send_interim_message(
                    chat_id,
                    "🔍 Analizando tu mensaje... Esto puede tardar hasta 1 minuto.",
                )
            )

        task = TextTask(
            query_id=query_id,
            user_hash=user_hash,
            chat_id=chat_id,
            text=text,
            timestamp=timestamp,
        )
        queue = os.getenv("RABBITMQ_QUEUE_TEXTS", "topic_texts")
        body = task.model_dump_json()
        logger.info(
            "route_message: publicando TextTask query_id=%s chat_id=%s queue=%s text_preview=%.60r",
            query_id, chat_id, queue, text,
        )

    else:
        logger.info(
            "route_message: tipo de mensaje no soportado (sin 'text' ni 'photo') - omitiendo. keys=%s",
            list(message.keys()),
        )
        messages_rejected.inc()
        return  # Tipo de payload no soportado (vídeo, sticker, etc.)

    # Comprobación de rate limit - después de validar el payload, antes de publicar.
    if not await check_rate_limit(user_hash):
        logger.warning(
            "route_message: rate limit superado - user_hash=%.12s... chat_id=%s",
            user_hash, chat_id,
        )
        messages_rejected.inc()
        if chat_id:
            await _send_interim_message(
                chat_id,
                "⏳ Has enviado demasiados mensajes. Por favor, espera 1 minuto antes de volver a consultar.",
            )
        return

    channel = await _get_channel()
    await channel.default_exchange.publish(
        aio_pika.Message(body=body.encode()),
        routing_key=queue,
    )
    if queue == os.getenv("RABBITMQ_QUEUE_TEXTS", "topic_texts"):
        texts_received.inc()
    else:
        images_received.inc()
    logger.info("route_message: mensaje publicado en queue=%s", queue)


async def publish_nlp_task(text: str) -> str:
    """Publica una tarea de texto directamente en la cola NLP y devuelve el task_id."""
    from shared.schemas import TextTask

    task = TextTask(
        user_hash="api_direct",
        chat_id=0,  # sin chat Telegram para llamadas directas a la API
        text=text,
        timestamp=datetime.now(timezone.utc),
    )
    task_id = str(task.query_id)
    queue = os.getenv("RABBITMQ_QUEUE_TEXTS", "topic_texts")

    channel = await _get_channel()
    await channel.default_exchange.publish(
        aio_pika.Message(body=task.model_dump_json().encode()),
        routing_key=queue,
    )
    return task_id
