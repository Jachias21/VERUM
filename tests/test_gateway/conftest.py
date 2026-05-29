"""
conftest.py — test_gateway

Registra services.gateway.app.* bajo el espacio de nombres ``app.*`` en
sys.modules durante cada módulo de test de gateway. Esto garantiza que
las importaciones internas de router.py (``from app.metrics import ...`` /
``from app.rate_limiter import ...``) encuentren los objetos del módulo de
gateway independientemente de qué otros conftest ya se hayan evaluado
durante la colección de pytest.

Contexto: pytest carga TODOS los conftest.py antes de ejecutar ningún test.
test_integration/conftest.py registraba app.* → módulos worker_nlp a nivel de
módulo (antes de cualquier test), lo que causaba fallos en los tests de gateway:
  ImportError: cannot import name 'texts_received' from 'services.worker_nlp.app.metrics'
El enfoque con fixture autouse aplaza el registro hasta que cada módulo de test
comienza realmente, y restaura sys.modules al terminar.
"""
from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture(autouse=True, scope="module")
def _gateway_app_modules():
    """Registra services.gateway.app.* como app.* durante la duración de este módulo."""
    _NAMES = ("metrics", "rate_limiter")
    saved: dict[str, object] = {}

    gw_pkg = importlib.import_module("services.gateway.app")
    saved["app"] = sys.modules.get("app")
    sys.modules["app"] = gw_pkg

    for name in _NAMES:
        key = f"app.{name}"
        saved[key] = sys.modules.get(key)
        sys.modules[key] = importlib.import_module(f"services.gateway.app.{name}")

    yield

    for key, original in saved.items():
        if original is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = original
