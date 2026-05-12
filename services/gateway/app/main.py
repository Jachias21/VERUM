"""
VERUM Gateway — FastAPI entry point.

Responsibilities:
  - Receive Telegram webhook POSTs and respond 200 OK immediately.
  - Delegate routing logic to router.py (no heavy work here).
"""
import asyncio
import logging
import os

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import Response
from pydantic import BaseModel
from dotenv import load_dotenv
from telegram import Bot
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.router import route_message, publish_nlp_task

load_dotenv()

app = FastAPI(title="VERUM Gateway", version="0.1.0")

logger = logging.getLogger(__name__)

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
    "• Recibirás un veredicto con fuente en menos de 15s\n\n"
    "⚠️ Textos muy cortos (menos de 50 caracteres) se ignoran."
)


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

    # Fire-and-forget: publish to queue and return 200 immediately
    await route_message(payload)
    return {"ok": True}


@app.post("/process_text/")
async def process_text(body: TextRequest):
    task_id = await publish_nlp_task(body.text)
    return {"task_id": task_id}
