"""
Configuracón de pytest para los tests de integración NLP.

Registra ``app.*`` como alias de ``services.worker_nlp.app.*`` en
``sys.modules`` *antes* de que se ejecute cada módulo de test de integración.
Esto garantiza:

1. Que las importaciones directas ``from app.xxx import ...`` de worker.py
   (diseñadas para Docker donde WORKDIR=/app == services/worker_nlp/app)
   funcionen al ejecutar desde la raíz del proyecto.

2. Que los objetos de módulo usados internamente por ``hybrid_search`` /
   ``synthesize_verdict`` y los apuntados por ``patch("services.worker_nlp.app.rag.*")``
   sean **idénticos** — evitando el problema de doble importación que
   haría invisibles los parches al código real de la función.

Nota de implementación: el registro se hace dentro de un fixture autouse
(no a nivel de módulo) para no contaminar sys.modules permanentemente durante
la colección de pytest. test_gateway/conftest.py usa el mismo patrón para
registrar los módulos de gateway, y el teardown/restore evita contaminación
cruzada entre suites de tests.
"""
from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture(autouse=True, scope="module")
def _worker_nlp_app_modules():
    """Registra services.worker_nlp.app.* como app.* durante la duración de este módulo."""
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
