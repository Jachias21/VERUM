"""
pytest configuration for NLP integration tests.

Registers ``app.*`` as aliases for ``services.worker_nlp.app.*`` in
``sys.modules`` *before* each integration test module runs.  This ensures:

1. worker.py's bare ``from app.xxx import ...`` imports (designed for Docker
   where WORKDIR=/app == services/worker_nlp/app) succeed when running from
   the project root.

2. The module objects used internally by ``hybrid_search`` / ``synthesize_verdict``
   and the module objects targeted by ``patch("services.worker_nlp.app.rag.*")``
   are **identical** — avoiding the double-import identity problem that would
   otherwise make patches invisible to the real function code.

Implementation note: registration is done inside an autouse fixture (not at
module level) so that it does not permanently pollute sys.modules during
pytest collection.  test_gateway/conftest.py uses the same pattern to register
gateway modules, and teardown/restore prevents cross-contamination between
test suites.
"""
from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture(autouse=True, scope="module")
def _worker_nlp_app_modules():
    """Register services.worker_nlp.app.* as app.* for the duration of this module."""
    _NAMES = ("cache", "metrics", "ner", "rag")
    saved: dict[str, object] = {}

    pkg = importlib.import_module("services.worker_nlp.app")
    saved["app"] = sys.modules.get("app")
    sys.modules["app"] = pkg

    for name in _NAMES:
        key = f"app.{name}"
        saved[key] = sys.modules.get(key)
        sys.modules[key] = importlib.import_module(f"services.worker_nlp.app.{name}")

    yield

    for key, original in saved.items():
        if original is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = original
