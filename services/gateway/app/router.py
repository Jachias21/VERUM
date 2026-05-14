"""
Message router — classifies incoming Telegram payloads and publishes
tasks to the appropriate RabbitMQ queue.
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

logger = logging.getLogger(__name__)


async def _send_interim_message(chat_id: int, text: str) -> None:
    """Send a fire-and-forget acknowledgment to the user via Telegram."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        logger.warning(
            "_send_interim_message: TELEGRAM_BOT_TOKEN not set — skipping interim to chat_id=%s",
            chat_id,
        )
        return
    try:
        bot = Bot(token=token)
        await bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "_send_interim_message: failed to send interim to chat_id=%s — %s: %s",
            chat_id, type(e).__name__, e,
        )


# Lazy singleton connection
_connection: aio_pika.abc.AbstractRobustConnection | None = None


async def _get_channel() -> aio_pika.abc.AbstractChannel:
    global _connection
    url = (
        f"amqp://{os.environ['RABBITMQ_USER']}:{os.environ['RABBITMQ_PASS']}"
        f"@{os.environ['RABBITMQ_HOST']}:{os.environ['RABBITMQ_PORT']}/"
    )
    if _connection is None or _connection.is_closed:
        _connection = await aio_pika.connect_robust(url)
    channel = await _connection.channel()
    # Declare queues as durable so messages survive broker restarts and are not
    # lost if the gateway starts before the workers.
    await channel.declare_queue(os.getenv("RABBITMQ_QUEUE_IMAGES", "topic_images"), durable=True)
    await channel.declare_queue(os.getenv("RABBITMQ_QUEUE_TEXTS", "topic_texts"), durable=True)
    return channel


async def route_message(payload: dict) -> None:
    message = payload.get("message") or payload.get("channel_post", {})
    if not message:
        logger.warning("route_message: payload contains no 'message' or 'channel_post' key — skipping. payload=%s", payload)
        messages_rejected.inc()
        return

    user_id = str(message.get("from", {}).get("id", "unknown"))
    user_hash = hashlib.sha256(user_id.encode()).hexdigest()
    chat_id: int = message.get("chat", {}).get("id", 0)
    query_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    if "photo" in message:
        from shared.schemas import ImageTask
        file_id = message["photo"][-1]["file_id"]   # highest resolution
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
                "route_message: text too short (%d chars, min=%d) — dropped. chat_id=%s",
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
        # Send interim acknowledgment (fire-and-forget) for non-command texts
        if not text.startswith("/"):
            asyncio.create_task(
                _send_interim_message(
                    chat_id,
                    "🔍 Analizando tu mensaje... Esto puede tardar hasta 15 segundos.",
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
            "route_message: publishing TextTask query_id=%s chat_id=%s queue=%s text_preview=%.60r",
            query_id, chat_id, queue, text,
        )

    else:
        logger.info(
            "route_message: unsupported message type (no 'text' or 'photo') — skipping. keys=%s",
            list(message.keys()),
        )
        messages_rejected.inc()
        return  # Unsupported payload type (video, sticker, etc.)

    # Rate limit check — applied after payload validation, before publishing.
    if not await check_rate_limit(user_hash):
        logger.warning(
            "route_message: rate limit exceeded — user_hash=%.12s… chat_id=%s",
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
    logger.info("route_message: message published to queue=%s", queue)


async def publish_nlp_task(text: str) -> str:
    """Publish a text task directly to the NLP queue and return the task_id."""
    from shared.schemas import TextTask

    task = TextTask(
        user_hash="api_direct",
        chat_id=0,  # no Telegram chat for direct API calls
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
