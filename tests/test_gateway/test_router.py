"""
Tests for services/gateway/app/router.py
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.schemas import ImageTask, TextTask


def _telegram_payload(text: str | None = None, photo: bool = False, chat_id: int = 111) -> dict:
    """Build a minimal Telegram update dict."""
    message: dict = {
        "from": {"id": 42},
        "chat": {"id": chat_id},
    }
    if text is not None:
        message["text"] = text
    if photo:
        message["photo"] = [
            {"file_id": "small_id", "width": 100, "height": 100},
            {"file_id": "large_id", "width": 1280, "height": 720},
        ]
    return {"message": message}


def _make_mock_channel() -> MagicMock:
    channel = AsyncMock()
    channel.default_exchange = AsyncMock()
    channel.default_exchange.publish = AsyncMock()
    channel.declare_queue = AsyncMock()
    return channel


# ── Test 14: short text is dropped, nothing published ────────────────────────

async def test_route_message_short_text_not_published():
    from services.gateway.app.router import route_message

    short_text = "Hola"  # definitely < 50 chars
    payload = _telegram_payload(text=short_text)

    mock_channel = _make_mock_channel()

    with (
        patch("services.gateway.app.router._get_channel", new=AsyncMock(return_value=mock_channel)),
        patch("services.gateway.app.router._send_interim_message", new=AsyncMock()),
    ):
        await route_message(payload)

    mock_channel.default_exchange.publish.assert_not_called()


# ── Test 15: long text publishes a valid TextTask ────────────────────────────

async def test_route_message_long_text_published():
    from services.gateway.app.router import route_message

    long_text = "El Gobierno de España anuncia que el nuevo virus detectado en Madrid no representa ningún peligro real."
    assert len(long_text) >= 60
    payload = _telegram_payload(text=long_text, chat_id=999)

    mock_channel = _make_mock_channel()

    with (
        patch("services.gateway.app.router._get_channel", new=AsyncMock(return_value=mock_channel)),
        patch("services.gateway.app.router._send_interim_message", new=AsyncMock()),
        patch("services.gateway.app.router.check_rate_limit", new=AsyncMock(return_value=True)),
    ):
        await route_message(payload)

    mock_channel.default_exchange.publish.assert_called_once()
    call_args = mock_channel.default_exchange.publish.call_args
    published_message = call_args[0][0]  # first positional arg to publish()
    body = json.loads(published_message.body.decode())

    # Must deserialize cleanly as a TextTask
    task = TextTask.model_validate(body)
    assert task.text == long_text
    assert task.chat_id == 999


# ── Test 16: photo payload publishes a valid ImageTask ───────────────────────

async def test_route_message_photo_published():
    from services.gateway.app.router import route_message

    payload = _telegram_payload(photo=True, chat_id=777)
    mock_channel = _make_mock_channel()

    with (
        patch("services.gateway.app.router._get_channel", new=AsyncMock(return_value=mock_channel)),
        patch("services.gateway.app.router._send_interim_message", new=AsyncMock()),
        patch("services.gateway.app.router.check_rate_limit", new=AsyncMock(return_value=True)),
    ):
        await route_message(payload)

    mock_channel.default_exchange.publish.assert_called_once()
    call_args = mock_channel.default_exchange.publish.call_args
    published_message = call_args[0][0]
    body = json.loads(published_message.body.decode())

    task = ImageTask.model_validate(body)
    assert task.telegram_file_id == "large_id"  # highest-resolution photo


# ── Test 17: unsupported payload type — nothing published, no exception ───────

async def test_route_message_unsupported_type_no_exception():
    from services.gateway.app.router import route_message

    # Sticker message — neither text nor photo
    payload = {"message": {"from": {"id": 1}, "chat": {"id": 2}, "sticker": {"file_id": "stk_1"}}}
    mock_channel = _make_mock_channel()

    with patch("services.gateway.app.router._get_channel", new=AsyncMock(return_value=mock_channel)):
        await route_message(payload)  # must not raise

    mock_channel.default_exchange.publish.assert_not_called()


# ── Test 18: rate limit blocks the 11th request ────────────────────────────────

async def test_rate_limit_blocks_after_max():
    from services.gateway.app.router import route_message

    long_text = "El Gobierno de España anuncia que el nuevo virus detectado en Madrid no representa ningún peligro real."
    payload = _telegram_payload(text=long_text, chat_id=123)

    mock_channel = _make_mock_channel()
    mock_interim = AsyncMock()

    with (
        patch("services.gateway.app.router._get_channel", new=AsyncMock(return_value=mock_channel)),
        patch("services.gateway.app.router._send_interim_message", new=mock_interim),
        patch("services.gateway.app.router.check_rate_limit", new=AsyncMock(return_value=False)),
    ):
        await route_message(payload)

    # Rate-limited: nothing published to the queue
    mock_channel.default_exchange.publish.assert_not_called()
    # User receives the courtesy rate-limit message
    mock_interim.assert_called_once()
    call_text = mock_interim.call_args[0][1]
    assert "⏳" in call_text or "demasiados" in call_text
