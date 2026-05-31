"""
conftest.py — test_gateway

Registers services.gateway.app.* under the bare ``app.*`` namespace in
sys.modules for the duration of each gateway test module.  This ensures
that router.py's internal ``from app.metrics import ...`` / ``from app.rate_limiter
import ...`` imports find the *gateway* module objects regardless of which
other conftest files have already been evaluated during pytest collection.

Background: pytest loads ALL conftest.py files before running any tests.
test_integration/conftest.py used to register app.* → worker_nlp modules at
module level (before any test ran), which meant gateway tests would fail with
  ImportError: cannot import name 'texts_received' from 'services.worker_nlp.app.metrics'
The autouse fixture approach defers registration until each test module
actually starts, and restores sys.modules on teardown.
"""
from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture(autouse=True, scope="module")
def _gateway_app_modules():
    """Register services.gateway.app.* as app.* for the duration of this module."""
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
