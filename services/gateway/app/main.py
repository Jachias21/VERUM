"""
VERUM Gateway — FastAPI entry point.

Responsibilities:
  - Receive Telegram webhook POSTs and respond 200 OK immediately.
  - Delegate routing logic to router.py (no heavy work here).
"""
import os

from fastapi import FastAPI, HTTPException, Header, Request
from dotenv import load_dotenv

from app.router import route_message

load_dotenv()

app = FastAPI(title="VERUM Gateway", version="0.1.0")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str = Header(default=None),
):
    if x_telegram_bot_api_secret_token != os.getenv("TELEGRAM_WEBHOOK_SECRET"):
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    payload = await request.json()

    # Fire-and-forget: publish to queue and return 200 immediately
    await route_message(payload)
    return {"ok": True}
