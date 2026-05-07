"""
Message router — classifies incoming Telegram payloads and publishes
tasks to the appropriate RabbitMQ queue.
"""
import os
import hashlib
import uuid
from datetime import datetime, timezone

import aio_pika

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
        text: str = message["text"]
        if len(text) < int(os.getenv("NLP_MIN_TEXT_LENGTH", 50)):
            return  # Too short to be a viral rumour
        task = TextTask(
            query_id=query_id,
            user_hash=user_hash,
            text=text,
            timestamp=timestamp,
        )
        queue = os.getenv("RABBITMQ_QUEUE_TEXTS", "topic_texts")
        body = task.model_dump_json()

    else:
        return  # Unsupported payload type (video, sticker, etc.)

    channel = await _get_channel()
    await channel.default_exchange.publish(
        aio_pika.Message(body=body.encode()),
        routing_key=queue,
    )


async def publish_nlp_task(text: str) -> str:
    """Publish a text task directly to the NLP queue and return the task_id."""
    from shared.schemas import TextTask

    task = TextTask(
        user_hash="api_direct",
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
