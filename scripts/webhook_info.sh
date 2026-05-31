#!/bin/sh
# =============================================================================
#  VERUM -- webhook_info.sh
#  Consulta el estado actual del webhook registrado en Telegram.
#  Uso: bash scripts/webhook_info.sh
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
# Consultar getWebhookInfo y mostrar de forma legible
# ---------------------------------------------------------------------------
"$PYTHON_CMD" -c "
import urllib.request, json, ssl, sys
token = sys.argv[1]
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
try:
    with urllib.request.urlopen(
        'https://api.telegram.org/bot{}/getWebhookInfo'.format(token),
        timeout=10, context=ctx
    ) as r:
        d = json.load(r)
except Exception as e:
    print('[!!] No se pudo conectar con la API de Telegram: ' + str(e))
    sys.exit(1)
if not d.get('ok'):
    print('[!!] Error de Telegram: ' + d.get('description', 'desconocido'))
    sys.exit(1)
r = d.get('result', {})
url        = r.get('url') or '(no registrado)'
pending    = str(r.get('pending_update_count', 0))
last_error = r.get('last_error_message', 'ninguno')
ip         = r.get('ip_address', 'desconocida')
print('')
print('--- Estado del Webhook Telegram --------------')
print('  URL configurada:    ' + url)
print('  Updates pendientes: ' + pending)
print('  Ultimo error:       ' + last_error)
print('  IP autorizada:      ' + ip)
print('-----------------------------------------------')
print('')
" "$TOKEN"
