"""Tests unitarios para shared.rabbitmq_utils."""
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_env(monkeypatch, overrides: dict) -> None:
    """Establece las cuatro vars de entorno AMQP requeridas, aplicando *overrides* encima."""
    defaults = {
        "RABBITMQ_USER": "verum",
        "RABBITMQ_PASS": "verum_pass",
        "RABBITMQ_HOST": "rabbitmq",
        "RABBITMQ_PORT": "5672",
    }
    defaults.update(overrides)
    for key, value in defaults.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


# ---------------------------------------------------------------------------
# build_amqp_url
# ---------------------------------------------------------------------------

def test_build_amqp_url_ok(monkeypatch):
    _set_env(monkeypatch, {})
    from shared.rabbitmq_utils import build_amqp_url

    url = build_amqp_url()
    assert url == "amqp://verum:verum_pass@rabbitmq:5672/"


def test_build_amqp_url_missing_var(monkeypatch):
    _set_env(monkeypatch, {"RABBITMQ_PASS": None})
    from shared.rabbitmq_utils import build_amqp_url

    with pytest.raises(RuntimeError, match="RABBITMQ_PASS"):
        build_amqp_url()


def test_build_amqp_url_encodes_special_chars(monkeypatch):
    _set_env(monkeypatch, {"RABBITMQ_PASS": "p@ss/word"})
    from shared.rabbitmq_utils import build_amqp_url

    url = build_amqp_url()
    # quote_plus codifica @ → %40, / → %2F
    assert "p%40ss%2Fword" in url
    # Los caracteres especiales en bruto NO deben aparecer en el segmento de credenciales
    assert ":p@ss/word@" not in url


# ---------------------------------------------------------------------------
# mask_amqp_url
# ---------------------------------------------------------------------------

def test_mask_amqp_url(monkeypatch):
    _set_env(monkeypatch, {"RABBITMQ_PASS": "s3cr3t!"})
    from shared.rabbitmq_utils import build_amqp_url, mask_amqp_url

    url = build_amqp_url()
    masked = mask_amqp_url(url)
    assert "s3cr3t" not in masked
    assert "***" in masked
    # Host and user should still be visible
    assert "verum" in masked
    assert "rabbitmq" in masked
