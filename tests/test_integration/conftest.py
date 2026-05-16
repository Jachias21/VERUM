"""
pytest configuration for NLP integration tests.

Registers ``app.*`` as aliases for ``services.worker_nlp.app.*`` in
``sys.modules`` *before* any test imports ``worker.py``.  This achieves two
things:

1. worker.py's bare ``from app.xxx import ...`` imports (designed for Docker
   where WORKDIR=/app == services/worker_nlp/app) succeed when running from
   the project root.

2. The module objects used internally by ``hybrid_search`` / ``synthesize_verdict``
   and the module objects targeted by ``patch("services.worker_nlp.app.rag.*")``
   are **identical** — avoiding the double-import identity problem that would
   otherwise make patches invisible to the real function code.
"""
from __future__ import annotations

import importlib
import sys


def _register_app_aliases() -> None:
    """Pre-import services.worker_nlp.app.* and expose them under the app.* key."""
    pkg = importlib.import_module("services.worker_nlp.app")
    sys.modules.setdefault("app", pkg)

    for name in ("cache", "metrics", "ner", "rag"):
        full = f"services.worker_nlp.app.{name}"
        mod = importlib.import_module(full)
        sys.modules.setdefault(f"app.{name}", mod)


_register_app_aliases()
