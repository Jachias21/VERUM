#!/bin/sh
# =============================================================================
#  VERUM -- webhook_delete.sh
#  Elimina el webhook actualmente registrado en Telegram.
#  Uso: bash scripts/webhook_delete.sh
# =============================================================================
set -e

ENV_FILE=".env"

# ---------------------------------------------------------------------------
# Leer el .env
# ---------------------------------------------------------------------------
if [ ! -f "$ENV_FILE" ]; then
  echo "[!!] Archivo .env no encontrado en la raiz del proyecto"
  exit 1
fi

TOKEN=$(grep -E "^TELEGRAM_BOT_TOKEN=" "$ENV_FILE" | cut -d'=' -f2- | tr -d '"' | tr -d "'")

if [ -z "$TOKEN" ]; then
  echo "[!!] TELEGRAM_BOT_TOKEN no configurado en .env"
  exit 1
fi

# ---------------------------------------------------------------------------
# Localizar Python (prioriza el venv del proyecto)
# ---------------------------------------------------------------------------
if [ -f "venv/Scripts/python.exe" ]; then
  PYTHON_CMD="venv/Scripts/python.exe"
elif [ -f "venv/bin/python" ]; then
  PYTHON_CMD="venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=python
else
  echo "[!!] Python no disponible. Instala Python 3 para usar este script."
  exit 1
fi

# ---------------------------------------------------------------------------
# Eliminar el webhook
# ---------------------------------------------------------------------------
echo "[>>] Eliminando webhook..."

"$PYTHON_CMD" -c "
import urllib.request, urllib.parse, json, ssl, sys
token  = sys.argv[1]
params = urllib.parse.urlencode({'drop_pending_updates': 'true'}).encode()
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
try:
    req = urllib.request.Request(
        'https://api.telegram.org/bot{}/deleteWebhook'.format(token),
        data=params, method='POST'
    )
    with urllib.request.urlopen(req, timeout=10, context=ctx) as r:
        d = json.load(r)
except Exception as e:
    print('[!!] No se pudo conectar con la API de Telegram: ' + str(e))
    sys.exit(1)
if d.get('ok'):
    print('[OK] Webhook eliminado correctamente.')
else:
    print('[!!] Error de Telegram: ' + d.get('description', 'desconocido'))
    sys.exit(1)
" "$TOKEN"
echo ""
