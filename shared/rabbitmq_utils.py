"""
Helpers de conexión AMQP compartidos.

API pública
----------
build_amqp_url() -> str
    Construye la URL completa ``amqp://`` a partir de variables de entorno,
    codificando las credenciales para que los caracteres especiales sean seguros.
    Lanza ``RuntimeError`` (no ``KeyError``) con el nombre de la variable ausente
    si alguna de las cuatro vars requeridas no está presente.

mask_amqp_url(url: str) -> str
    Devuelve una copia de *url* con la contraseña sustituida por ``***`` para
    poder escribir la URL en logs sin filtrar credenciales.
"""
import os
import re
from urllib.parse import quote_plus

_REQUIRED_VARS = ("RABBITMQ_USER", "RABBITMQ_PASS", "RABBITMQ_HOST", "RABBITMQ_PORT")


def build_amqp_url() -> str:
    """Devuelve ``amqp://<user>:<pass>@<host>:<port>/`` con credenciales codificadas en URL.

    Raises
    ------
    RuntimeError
        Si alguna de las cuatro variables de entorno requeridas no está definida.
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
    """Devuelve *url* con el segmento de contraseña reemplazado por ``***``.

    Funciona con cualquier URL estilo ``amqp://user:password@host``.
    """
    return re.sub(r"(amqps?://[^:]+:)[^@]+(@)", r"\1***\2", url)
