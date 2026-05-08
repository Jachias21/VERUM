"""
Message router — classifies incoming Telegram payloads and publishes
tasks to the appropriate RabbitMQ queue.
"""
import os
import hashlib
import logging
import uuid
from datetime import datetime, timezone

import aio_pika

logger = logging.getLogger(__name__)

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
    return await _connection.channel()


async def route_message(payload: dict) -> None:
    message = payload.get("message") or payload.get("channel_post", {})
    if not message:
        logger.warning("route_message: payload contains no 'message' or 'channel_post' key — skipping. payload=%s", payload)
        return

    user_id = str(message.get("from", {}).get("id", "unknown"))
    user_hash = hashlib.sha256(user_id.encode()).hexdigest()
    query_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    if "photo" in message:
        from shared.schemas import ImageTask
        file_id = message["photo"][-1]["file_id"]   # highest resolution
        task = ImageTask(
            query_id=query_id,
            user_hash=user_hash,
            telegram_file_id=file_id,
            timestamp=timestamp,
        )
        queue = os.getenv("RABBITMQ_QUEUE_IMAGES", "topic_images")
        body = task.model_dump_json()

    elif "text" in message:
        from shared.schemas import TextTask
        text: str = message.get("text", "")
        chat_id: int = message.get("chat", {}).get("id", 0)
        min_length = int(os.getenv("NLP_MIN_TEXT_LENGTH", 3))
        if len(text) < min_length:
            logger.info(
                "route_message: text too short (%d chars, min=%d) — dropped. chat_id=%s",
                len(text), min_length, chat_id,
            )
            return
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
        return  # Unsupported payload type (video, sticker, etc.)

    channel = await _get_channel()
    await channel.default_exchange.publish(
        aio_pika.Message(body=body.encode()),
        routing_key=queue,
    )
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
