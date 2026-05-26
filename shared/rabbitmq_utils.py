"""
Shared AMQP connection helpers.

Public API
----------
build_amqp_url() -> str
    Builds the full ``amqp://`` URL from environment variables, URL-encoding
    credentials so that special characters (``@``, ``:``, ``/``) are safe.
    Raises ``RuntimeError`` (not ``KeyError``) with the missing variable name
    if any of the four required vars is absent.

mask_amqp_url(url: str) -> str
    Returns a copy of *url* with the password replaced by ``***`` so that the
    URL can be written to logs without leaking credentials.
"""
import os
import re
from urllib.parse import quote_plus

_REQUIRED_VARS = ("RABBITMQ_USER", "RABBITMQ_PASS", "RABBITMQ_HOST", "RABBITMQ_PORT")


def build_amqp_url() -> str:
    """Return ``amqp://<user>:<pass>@<host>:<port>/`` with URL-encoded credentials.

    Raises
    ------
    RuntimeError
        If any of the four required environment variables is not set.
    """
    values: dict[str, str] = {}
    for var in _REQUIRED_VARS:
        val = os.environ.get(var)
        if val is None:
            raise RuntimeError(f"Missing required env var: {var}")
        values[var] = val

    user = quote_plus(values["RABBITMQ_USER"])
    password = quote_plus(values["RABBITMQ_PASS"])
    host = values["RABBITMQ_HOST"]
    port = values["RABBITMQ_PORT"]
    return f"amqp://{user}:{password}@{host}:{port}/"


def mask_amqp_url(url: str) -> str:
    """Return *url* with the password segment replaced by ``***``.

    Works on any ``amqp://user:password@host`` style URL.
    """
    return re.sub(r"(amqps?://[^:]+:)[^@]+(@)", r"\1***\2", url)
