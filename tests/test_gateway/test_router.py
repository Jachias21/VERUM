"""
Tests para services/gateway/app/router.py
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("aio_pika", reason="aio_pika not installed (Docker-only dependency)")

from shared.schemas import ImageTask, TextTask


def _telegram_payload(text: str | None = None, photo: bool = False, chat_id: int = 111) -> dict:
    """Construye un dict mínimo de update de Telegram."""
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


# ── Test 14: texto corto descartado, nada publicado ─────────────────────────

async def test_route_message_short_text_not_published():
    from services.gateway.app.router import route_message

    short_text = "Hola"  # definitivamente < 50 chars
    payload = _telegram_payload(text=short_text)

    mock_channel = _make_mock_channel()

    with (
        patch("services.gateway.app.router._get_channel", new=AsyncMock(return_value=mock_channel)),
        patch("services.gateway.app.router._send_interim_message", new=AsyncMock()),
    ):
        await route_message(payload)

    mock_channel.default_exchange.publish.assert_not_called()


# ── Test 15: texto largo publica un TextTask válido ─────────────────────────

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
    published_message = call_args[0][0]  # primer argumento posicional de publish()
    body = json.loads(published_message.body.decode())

    # Debe deserializarse correctamente como TextTask
    task = TextTask.model_validate(body)
    assert task.text == long_text
    assert task.chat_id == 999


# ── Test 16: payload de foto publica un ImageTask válido ────────────────────

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
    assert task.telegram_file_id == "large_id"  # foto de mayor resolución


# ── Test 17: tipo de payload no soportado — nada publicado, sin excepción ───

async def test_route_message_unsupported_type_no_exception():
    from services.gateway.app.router import route_message

    # Mensaje con sticker — ni texto ni foto
    payload = {"message": {"from": {"id": 1}, "chat": {"id": 2}, "sticker": {"file_id": "stk_1"}}}
    mock_channel = _make_mock_channel()

    with patch("services.gateway.app.router._get_channel", new=AsyncMock(return_value=mock_channel)):
        await route_message(payload)  # no debe lanzar excepción

    mock_channel.default_exchange.publish.assert_not_called()


# ── Test 18: rate limit bloquea la undecima petición ─────────────────────────

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

    # Rate-limited: nada publicado en la cola
    mock_channel.default_exchange.publish.assert_not_called()
    # El usuario recibe el mensaje de rate-limit.
    # NOTA: route_message también agenda un interim de análisis vía asyncio.create_task
    # antes de llegar al check de rate-limit, por lo que el mock puede llamarse 1 o 2 veces.
    assert mock_interim.called, "_send_interim_message no fue llamado"
    all_texts = [c[0][1] for c in mock_interim.call_args_list]
    assert any("⏳" in t or "demasiados" in t for t in all_texts), (
        f"Expected rate-limit message in calls, got: {all_texts}"
    )
