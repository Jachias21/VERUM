"""
VERUM Gateway — FastAPI entry point.

Responsibilities:
  - Receive Telegram webhook POSTs and respond 200 OK immediately.
  - Delegate routing logic to router.py (no heavy work here).
"""
import asyncio
import hashlib
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import Response
from pydantic import BaseModel
from dotenv import load_dotenv
from telegram import Bot
from telegram.error import TelegramError
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.router import route_message, publish_nlp_task
from shared.db import get_mongo_db

load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ──────────────────────────────────────────────────────────────
    webhook_base = os.getenv("WEBHOOK_BASE_URL", "").strip()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    secret = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()

    if webhook_base and token:
        webhook_url = f"{webhook_base}/webhook"
        try:
            bot = Bot(token=token)
            await asyncio.wait_for(
                bot.set_webhook(url=webhook_url, secret_token=secret or None),
                timeout=5,
            )
            logger.info("[gateway] Webhook registrado en %s", webhook_url)
        except asyncio.TimeoutError:
            logger.error("[gateway] set_webhook timeout — Telegram no respondió en 5 s")
        except TelegramError as exc:
            logger.error("[gateway] set_webhook falló — %s: %s", type(exc).__name__, exc)
    else:
        logger.info("[gateway] WEBHOOK_BASE_URL no configurada — registro automático omitido")

    yield

    # ── shutdown ──────────────────────────────────────────────────────────────
    if token:
        try:
            bot = Bot(token=token)
            await asyncio.wait_for(bot.delete_webhook(), timeout=5)
            logger.info("[gateway] Webhook eliminado")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[gateway] delete_webhook falló — %s", exc)


app = FastAPI(title="VERUM Gateway", version="0.1.0", lifespan=lifespan)

_START_TEXT = (
    "🛡️ *Bienvenido a VERUM*\n\n"
    "Soy tu perito forense digital. Envíame cualquier texto viral o noticia sospechosa "
    "y te diré si es FAKE, REAL o UNVERIFIED, citando la fuente.\n\n"
    "📖 Usa /help para más info."
)

_HELP_TEXT = (
    "ℹ️ *Cómo usar VERUM*\n\n"
    "• Envíame cualquier texto viral o cadena de WhatsApp\n"
    "• Analizaré las entidades clave y buscaré desmentidos\n"
    "• Recibirás un veredicto con fuente en menos de 15s\n"
    "• Usa /feedback correcto o /feedback incorrecto después de un análisis para ayudarnos a mejorar\n\n"
    "⚠️ Textos muy cortos (menos de 50 caracteres) se ignoran."
)


_FEEDBACK_USAGE = (
    "📋 *Uso:* `/feedback correcto` o `/feedback incorrecto`\n"
    "Úsalo justo después de recibir un veredicto para ayudarnos a mejorar."
)

_FEEDBACK_POSITIVE = "¡Gracias! Tu feedback ayuda a mejorar VERUM."
_FEEDBACK_NEGATIVE = "Gracias por reportarlo. Revisaremos este caso."
_FEEDBACK_NOT_FOUND = (
    "⚠️ No encontré ningún análisis reciente tuyo. "
    "Envía un texto primero y luego usa /feedback."
)


async def _handle_feedback(chat_id: int, user_id: str, argument: str) -> None:
    """Process /feedback command: update the latest QueryLog for this user."""
    argument = argument.strip().lower()

    if argument not in ("correcto", "incorrecto"):
        await _send_command_reply(chat_id, _FEEDBACK_USAGE)
        return

    feedback_value = "correct" if argument == "correcto" else "incorrect"
    user_hash = hashlib.sha256(user_id.encode()).hexdigest()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=2)

    try:
        db = get_mongo_db()
        collection = db[os.getenv("MONGO_COLLECTION_QUERIES", "queries")]
        doc = await collection.find_one(
            {"user_hash": user_hash, "timestamp": {"$gte": cutoff}},
            sort=[("timestamp", -1)],
        )
        if doc is None:
            await _send_command_reply(chat_id, _FEEDBACK_NOT_FOUND)
            return
        await collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"feedback": feedback_value}},
        )
        reply = _FEEDBACK_POSITIVE if feedback_value == "correct" else _FEEDBACK_NEGATIVE
        await _send_command_reply(chat_id, reply)
    except Exception as exc:  # noqa: BLE001
        logger.error("_handle_feedback: MongoDB error — %s: %s", type(exc).__name__, exc)
        await _send_command_reply(chat_id, "⚠️ Error al guardar el feedback. Inténtalo de nuevo.")


async def _send_command_reply(chat_id: int, text: str) -> None:
    """Send a reply to a bot command via Telegram, silently ignoring failures."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        logger.warning("_send_command_reply: TELEGRAM_BOT_TOKEN not set — skipping reply to chat_id=%s", chat_id)
        return
    try:
        bot = Bot(token=token)
        await bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
    except Exception as e:  # noqa: BLE001
        logger.warning("_send_command_reply: failed to send to chat_id=%s — %s: %s", chat_id, type(e).__name__, e)


class TextRequest(BaseModel):
    text: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str = Header(default=None),
):
    if x_telegram_bot_api_secret_token != os.getenv("TELEGRAM_WEBHOOK_SECRET"):
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    payload = await request.json()

    message = payload.get("message") or payload.get("channel_post") or {}
    text = message.get("text", "")
    chat_id = message.get("chat", {}).get("id", 0)

    if text == "/start":
        asyncio.create_task(_send_command_reply(chat_id, _START_TEXT))
        return {"ok": True}

    if text == "/help":
        asyncio.create_task(_send_command_reply(chat_id, _HELP_TEXT))
        return {"ok": True}

    if text.startswith("/feedback"):
        parts = text.split(maxsplit=1)
        argument = parts[1] if len(parts) > 1 else ""
        user_id = str(message.get("from", {}).get("id", ""))
        if not argument:
            asyncio.create_task(_send_command_reply(chat_id, _FEEDBACK_USAGE))
        else:
            asyncio.create_task(_handle_feedback(chat_id, user_id, argument))
        return {"ok": True}

    # Fire-and-forget: publish to queue and return 200 immediately
    await route_message(payload)
    return {"ok": True}


@app.post("/process_text/")
async def process_text(body: TextRequest):
    task_id = await publish_nlp_task(body.text)
    return {"task_id": task_id}
