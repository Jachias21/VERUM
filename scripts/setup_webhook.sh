#!/bin/sh
# =============================================================================
#  VERUM -- setup_webhook.sh
#  Detecta la URL publica de ngrok y registra el webhook con Telegram.
#  Uso: bash scripts/setup_webhook.sh
# =============================================================================
set -e

ENV_FILE=".env"

echo ""
echo "============================================"
echo "  VERUM -- Webhook Telegram"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# 1. Leer el .env
# ---------------------------------------------------------------------------
if [ ! -f "$ENV_FILE" ]; then
  echo "[!!] Archivo .env no encontrado en la raiz del proyecto"
  exit 1
fi

TOKEN=$(grep -E "^TELEGRAM_BOT_TOKEN=" "$ENV_FILE" | cut -d'=' -f2- | tr -d '"' | tr -d "'")
SECRET=$(grep -E "^TELEGRAM_WEBHOOK_SECRET=" "$ENV_FILE" | cut -d'=' -f2- | tr -d '"' | tr -d "'")

if [ -z "$TOKEN" ]; then
  echo "[!!] TELEGRAM_BOT_TOKEN no configurado en .env"
  exit 1
fi

if [ -z "$SECRET" ]; then
  echo "[!!] TELEGRAM_WEBHOOK_SECRET no configurado en .env"
  exit 1
fi

# ---------------------------------------------------------------------------
# 2+3. Detectar Python disponible y usarlo para consultar la API de ngrok
#      (evita el pipe curl | python que falla en Windows con el stub del Store)
# ---------------------------------------------------------------------------
echo "[>>] Detectando URL de ngrok..."

# Localizar Python: prioriza el venv del proyecto, luego el PATH del sistema
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

# Un solo script Python que escanea puertos 4040-4043 y extrae la URL HTTPS
NGROK_RESULT=$("$PYTHON_CMD" -c "
import urllib.request, json, sys

def get_url(port):
    try:
        with urllib.request.urlopen(
            'http://localhost:{}/api/tunnels'.format(port), timeout=3
        ) as r:
            data = json.load(r)
        tunnels = data.get('tunnels', [])
        # 1) URL empieza por https
        urls = [t.get('public_url','') for t in tunnels
                if t.get('public_url','').startswith('https')]
        # 2) proto == https
        if not urls:
            urls = [t.get('public_url','') for t in tunnels
                    if t.get('proto','') == 'https']
        # 3) cualquier tunel, convertir a https
        if not urls:
            all_urls = [t.get('public_url','') for t in tunnels
                        if t.get('public_url','')]
            if all_urls:
                urls = [all_urls[0].replace('http://','https://',1)]
        return urls[0] if urls else None
    except Exception:
        return None

for p in [4040, 4041, 4042, 4043]:
    url = get_url(p)
    if url:
        print(url)
        sys.exit(0)

# Nada encontrado: distinguir entre ngrok apagado y sin tuneles HTTPS
try:
    urllib.request.urlopen('http://localhost:4040/api/tunnels', timeout=3)
    print('__NO_HTTPS__')
except Exception:
    print('__DOWN__')
" 2>/dev/null || echo "__DOWN__")

case "$NGROK_RESULT" in
  __DOWN__)
    echo "[!!] ngrok no esta corriendo. Inicialo con: ngrok http 8000"
    exit 1
    ;;
  __NO_HTTPS__)
    echo "[!!] ngrok esta corriendo pero no hay tuneles HTTPS activos"
    exit 1
    ;;
  "")
    echo "[!!] ngrok no esta corriendo. Inicialo con: ngrok http 8000"
    exit 1
    ;;
  *)
    NGROK_URL="$NGROK_RESULT"
    ;;
esac
echo "[OK] URL detectada: $NGROK_URL"

# ---------------------------------------------------------------------------
# 4. Registrar el webhook con Telegram
# ---------------------------------------------------------------------------
echo "[>>] Registrando webhook con Telegram..."

TELEGRAM_RESP=$("$PYTHON_CMD" -c "
import urllib.request, urllib.parse, json, ssl, sys
token  = sys.argv[1]
url    = sys.argv[2]
secret = sys.argv[3]
params = urllib.parse.urlencode({
    'url': url + '/webhook',
    'secret_token': secret,
    'drop_pending_updates': 'true'
}).encode()
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
try:
    req = urllib.request.Request(
        'https://api.telegram.org/bot{}/setWebhook'.format(token),
        data=params, method='POST'
    )
    with urllib.request.urlopen(req, timeout=10, context=ctx) as r:
        print(r.read().decode())
except Exception as e:
    print('')
" "$TOKEN" "$NGROK_URL" "$SECRET" 2>/dev/null || true)

if [ -z "$TELEGRAM_RESP" ]; then
  echo "[!!] No se pudo conectar con la API de Telegram. Verifica tu conexion."
  exit 1
fi

TG_OK=$("$PYTHON_CMD" -c "
import json, sys
d = json.loads(sys.argv[1])
print('yes' if d.get('ok') else 'no:' + d.get('description','error desconocido'))
" "$TELEGRAM_RESP" 2>/dev/null || echo "no:parse_error")

case "$TG_OK" in
  yes)
    echo "[OK] Webhook registrado correctamente."
    ;;
  no:*)
    echo "[!!] Error de Telegram: ${TG_OK#no:}"
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# 5. Actualizar WEBHOOK_BASE_URL en el .env
# ---------------------------------------------------------------------------
echo "[>>] Actualizando .env..."

# Hacer backup
cp "$ENV_FILE" "${ENV_FILE}.bak"

if grep -qE "^WEBHOOK_BASE_URL=" "$ENV_FILE"; then
  # La linea ya existe: reemplazarla con Python (sed -i difiere entre GNU y BSD/macOS)
  "$PYTHON_CMD" -c "
import re, sys
path, new_val = sys.argv[1], sys.argv[2]
with open(path, 'r') as f:
    content = f.read()
content = re.sub(r'^WEBHOOK_BASE_URL=.*$', 'WEBHOOK_BASE_URL=' + new_val, content, flags=re.MULTILINE)
with open(path, 'w') as f:
    f.write(content)
" "$ENV_FILE" "$NGROK_URL"
else
  # La linea no existe: anadirla al final
  printf '\nWEBHOOK_BASE_URL=%s\n' "$NGROK_URL" >> "$ENV_FILE"
fi

echo "[OK] WEBHOOK_BASE_URL actualizada en .env"
echo ""
echo "El bot esta listo para recibir mensajes."
echo ""
