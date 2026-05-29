# VERUM

> *"En un mundo donde la información se manipula, VERUM transforma datos en evidencia para revelar la verdad."*

Bot de Telegram que actúa como perito forense digital: analiza imágenes y textos virales para determinar si son falsos, generados por IA o desinformación verificada. Proyecto TFM del Máster en IA & Big Data.

---

## Índice

1. [¿Qué hace VERUM?](#qué-hace-verum)
2. [Arquitectura general](#arquitectura-general)
3. [Estructura del repositorio](#estructura-del-repositorio)
4. [Servicios en detalle](#servicios-en-detalle)
5. [Módulos de ML](#módulos-de-ml)
6. [Infraestructura de datos](#infraestructura-de-datos)
7. [Puesta en marcha](#puesta-en-marcha)
8. [Variables de entorno](#variables-de-entorno)

---

## ¿Qué hace VERUM?

El usuario reenvía al bot de Telegram una imagen sospechosa o un texto viral. VERUM responde en menos de 15 segundos con:

- **Para imágenes:** veredicto REAL / FAKE + mapa de calor (Grad-CAM) que señala visualmente las zonas con artefactos sintéticos.
- **Para textos:** veredicto REAL / FAKE / UNVERIFIED + resumen de 3 líneas redactado por un LLM local citando la fuente de fact-checking.

Todo el procesamiento ocurre en servidores propios. Ningún dato del usuario sale a APIs de terceros.

---

## Arquitectura general

```
                        ┌─────────────────────────────────────┐
                        │           Usuario (Telegram)        │
                        └──────────────┬──────────────────────┘
                                       │ envía imagen o texto
                                       ▼
                        ┌─────────────────────────────────────┐
                        │         Gateway  (FastAPI)          │
                        │   /webhook  ──►  router.py          │
                        │   Responde 200 OK de inmediato      │
                        └───────────┬─────────────────────────┘
                                    │ publica tarea en cola
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          topic_images (RabbitMQ)        topic_texts (RabbitMQ)
                    │                               │
                    ▼                               ▼
     ┌──────────────────────────┐    ┌──────────────────────────┐
     │     worker_vision        │    │      worker_nlp          │
     │                          │    │                          │
     │  1. Descarga imagen      │    │  1. SpaCy NER            │
     │  2. RGB → YCbCr → DFT    │    │  2. Búsqueda Qdrant      │
     │  3. CNN Two-Stream ONNX  │    │  3. Fallback Google API  │
     │  4. Grad-CAM heatmap     │    │  4. Síntesis con Ollama  │
     └──────────┬───────────────┘    └────────────┬─────────────┘
                │                                 │
                └──────────────┬──────────────────┘
                               │ guarda metadatos
                               ▼
                    ┌─────────────────────┐
                    │      MongoDB        │  ◄── Dashboard (Streamlit)
                    │  colección: queries │
                    └─────────────────────┘

                    ┌─────────────────────┐
                    │       Qdrant        │  ◄── ETL pipeline (cron)
                    │  knowledge base     │
                    └─────────────────────┘

                    ┌─────────────────────┐
                    │       Ollama        │
                    │  Llama 3.2 / Qwen   │
                    └─────────────────────┘
```

### Principio de diseño clave: desacoplamiento asíncrono

El Gateway **nunca** procesa nada pesado. Recibe el mensaje, lo mete en una cola RabbitMQ y devuelve `200 OK` a Telegram en milisegundos. Los Workers consumen la cola a su propio ritmo (`prefetch=1`), garantizando que la RAM/VRAM nunca se sature aunque lleguen 100 peticiones simultáneas.

---

## Estructura del repositorio

```
VERUM/
│
├── services/                  # Microservicios deployables (cada uno con su Dockerfile)
│   ├── gateway/               # Punto de entrada HTTP — recibe webhooks de Telegram
│   ├── worker_vision/         # Worker de análisis forense de imágenes
│   ├── worker_nlp/            # Worker de fact-checking de textos
│   └── etl/                   # Pipeline de actualización de la base de conocimiento
│
├── models/                    # Código de entrenamiento offline (NO se despliega en Docker)
│   ├── vision/                # Arquitectura CNN, entrenamiento, evaluación y export ONNX
│   └── nlp/                   # Utilidades de embeddings (BAAI/bge-m3)
│
├── shared/                    # Paquete Python compartido entre todos los servicios
│   ├── schemas.py             # Contratos de mensajes RabbitMQ (Pydantic)
│   └── db.py                  # Factories de clientes MongoDB y Qdrant
│
├── dashboard/                 # Dashboard analítico (Streamlit)
├── landing/                   # Web pública del proyecto (React/Tailwind)
│
├── data/
│   ├── raw/                   # Datasets originales (gitignored)
│   ├── processed/             # Datasets preprocesados para entrenamiento (gitignored)
│   └── knowledge_base/        # Artículos de fact-checking descargados por el ETL
│
├── notebooks/                 # Jupyter notebooks de exploración y prototipado
├── tests/                     # Tests por servicio
│
├── docker-compose.yml         # Orquestación completa de producción
├── docker-compose.dev.yml     # Overrides para desarrollo (hot-reload, bind mounts)
├── .env.example               # Plantilla de variables de entorno
└── Makefile                    # Comandos útiles (registro webhook, tests, etc.)
```

---

## Servicios en detalle

### `services/gateway/` — API Gateway

**Tecnología:** FastAPI + aio-pika
**Puerto:** `8000`

El único punto de entrada del sistema. Expone un único endpoint `POST /webhook` que valida la firma de Telegram y delega inmediatamente en `router.py`.

```
app/
├── main.py      # Arranque FastAPI, endpoint /webhook y /health
└── router.py    # Clasifica el payload (imagen vs texto), hashea el user_id
                 # y publica un ImageTask o TextTask en RabbitMQ
```

**Flujo:**
1. Telegram envía un `POST /webhook` con el mensaje.
2. Se valida el header `X-Telegram-Bot-Api-Secret-Token`.
3. `router.py` extrae el `file_id` (imágenes) o el `text` (textos largos).
4. Genera un `query_id` UUID, hashea el `user_id` con SHA-256 (privacidad RGPD).
5. Publica el payload serializado como JSON en la cola correspondiente.
6. Devuelve `{"ok": true}` — Telegram recibe la respuesta en < 100ms.

---

### `services/worker_vision/` — Worker de Visión

**Tecnología:** onnxruntime + OpenCV + SciPy + pytorch-grad-cam

Consume mensajes de `topic_images`. Para cada imagen detecta si es real (cámara física) o sintética (IA generativa).

```
app/
├── worker.py      # Bucle RabbitMQ: consume, orquesta, loguea y responde
├── inference.py   # Pipeline completo de preprocesado + inferencia ONNX
└── xai.py         # Grad-CAM sobre la rama espacial → heatmap PNG
```

**Pipeline de `inference.py`:**
1. Decodifica bytes → imagen BGR (OpenCV).
2. Convierte a YCbCr, aísla canales Cb y Cr.
3. Aplica DFT 2D + filtro paso alto → tensor de frecuencias (2 canales).
4. Carga modelo ONNX (`VISION_MODEL_PATH`) y ejecuta las dos ramas simultáneamente:
   - Rama espacial: imagen RGB original.
   - Rama frecuencial: espectro DFT de Cb/Cr.
5. Devuelve `ai_confidence_score` (0 = cámara real, 1 = IA sintética).

**Si el veredicto es FAKE:** `xai.py` genera un Grad-CAM sobre la rama espacial y lo superpone a la imagen original. El mapa de calor se adjunta a la respuesta de Telegram.

> El modelo se entrena en `models/vision/` y se exporta a ONNX antes de desplegar. El contenedor de producción usa sólo `onnxruntime`, no PyTorch.

---

### `services/worker_nlp/` — Worker de NLP

**Tecnología:** SpaCy + LangChain + Qdrant + Ollama (Llama 3.2 / Qwen 2.5) + Prometheus  
**Puerto de métricas:** `9101` (`/metrics`)

Consume mensajes de `topic_texts`. Para cada texto viral extrae entidades, busca desmentidos y genera un veredicto explicado. Expone métricas Prometheus scrapeables en `http://localhost:9101/metrics`, incluyendo el histograma `verum_nlp_processing_seconds` (latencia extremo a extremo por mensaje).

```
app/
├── worker.py    # Bucle RabbitMQ: consume, orquesta, loguea y responde
├── ner.py       # Limpieza de texto + extracción de entidades (SpaCy es_core_news_lg)
├── rag.py       # Pipeline RAG híbrido en dos niveles + síntesis LLM
├── cache.py     # Caché de veredictos en MongoDB (evita re-procesar duplicados)
└── metrics.py   # Histograma Prometheus: verum_nlp_processing_seconds
```

**Pipeline de `rag.py` (dos niveles):**

```
Nivel 1 — Local (baja latencia, sin coste):
  Búsqueda Híbrida en Qdrant
    ├── Vectores densos  (BAAI/bge-m3, similitud semántica)
    └── Vectores dispersos BM25 (coincidencia exacta de keywords)
  → Si score ≥ NLP_CONFIDENCE_THRESHOLD: usar resultado local

Nivel 2 — Fallback online (bulo de "Día Cero"):
  ├── Google Fact Check Tools API
  └── GNews API (en paralelo)
  Ambas buscan por las entidades extraídas; resultados fusionados con RRF

Fusión: RRF (Reciprocal Rank Fusion) de todas las listas
Síntesis: prompt al LLM local vía Ollama → veredicto en 3 líneas
```

---

### `services/etl/` — Pipeline ETL

**Tecnología:** feedparser + sentence-transformers + qdrant-client  
**Ejecución:** automática (`etl_scheduler`) o manual (`--profile etl`)

Mantiene actualizada la base de conocimiento local (Qdrant) con los últimos artículos de fact-checking.

```
app/
└── pipeline.py   # Extract (RSS feeds) → Transform (embeddings) → Load (Qdrant upsert)
```

Dos modos de ejecución coexisten:
- **`etl_scheduler`** — Servicio que arranca automáticamente con `docker compose up -d`. Ejecuta el pipeline según la expresión cron configurada en `ETL_SCHEDULE` (por defecto `0 */12 * * *`, cada 12 horas). El primer ciclo se lanza inmediatamente al arrancar el contenedor.
- **`etl` (manual)** — Para lanzamientos puntuales: `docker compose --profile etl run etl`.

**Fuentes configuradas por defecto:** Maldita.es · Newtral.es · Snopes.com  
Se pueden añadir más URLs en `RSS_FEEDS` dentro de `pipeline.py`.

---

### `dashboard/` — Dashboard analítico

**Tecnología:** Streamlit + Plotly + PyMongo
**Puerto:** `8501`

Se conecta a MongoDB en tiempo real y muestra:
- Total de consultas, FakeNews detectadas, latencia media, usuarios únicos.
- Distribución de veredictos (pie chart).
- Evolución temporal de consultas (line chart).
- Nube de palabras (word cloud) con las entidades más consultadas de la última semana, en paleta teal/cyan (usa todos los datos disponibles si hay menos de 5 consultas recientes).
- Top 20 entidades extraídas (bar chart complementario).

---

### `landing/` — Web pública

Página promocional del proyecto. Explica el funcionamiento, muestra demos de Grad-CAM y proporciona el enlace directo al bot (`tg://resolve?domain=VerumBot`).

**Stack recomendado:** React + Vite + Tailwind CSS
**Despliegue:** Vercel / Netlify / GitHub Pages (CI/CD automático desde `main`)

---

## Módulos de ML

Estos módulos se usan **offline** (entrenamiento en local/home-lab con GPU AMD ROCm). El resultado final se exporta a ONNX y se copia en `models/vision/weights/` para que `worker_vision` lo consuma en producción.

### `models/vision/`

| Fichero | Qué contiene |
|---|---|
| `architecture.py` | Clase `TwoStreamCNN`: rama espacial EfficientNet-B0 + rama frecuencial CNN custom, fusión en capas densas |
| `train.py` | Bucle de entrenamiento completo con data augmentation de compresión JPEG (simula Telegram) |
| `evaluate.py` | Métricas (Accuracy, F1, AUC) + exportación a ONNX |
| `weights/` | Modelos entrenados `.pt` y `.onnx` (gitignored, sólo `.gitkeep`) |

**Dataset esperado:**
```
data/processed/
├── train/real/   ← Fotos de cámara (RAISE, MS COCO)
├── train/fake/   ← Imágenes IA (ArtiFact, CIFAKE, generaciones propias)
├── val/real/
├── val/fake/
├── test/real/
└── test/fake/
```

### `models/nlp/`

| Fichero | Qué contiene |
|---|---|
| `embeddings.py` | Singleton de `SentenceTransformer(BAAI/bge-m3)` compartido entre ETL y worker_nlp |

---

## Infraestructura de datos

### RabbitMQ — Cola de mensajes

| Cola | Productor | Consumidor |
|---|---|---|
| `topic_images` | gateway | worker_vision |
| `topic_texts` | gateway | worker_nlp |

UI de gestión accesible en `http://localhost:15672` (user/pass en `.env`).

### MongoDB — Almacenamiento analítico

Colección `queries`. Cada documento registra los metadatos de una interacción:

```json
{
  "query_id": "uuid",
  "timestamp": "ISO-8601",
  "user_hash": "sha256(telegram_user_id)",
  "payload_type": "image | text",
  "total_processing_time_ms": 4230,
  "final_verdict": "FAKE | REAL | UNVERIFIED",

  // Si imagen:
  "image_resolution": "1024x768",
  "ai_confidence_score": 0.94,
  "prnu_detected": false,

  // Si texto:
  "extracted_entities": ["hacker", "WhatsApp", "Policía Nacional"],
  "fact_check_matches": 3,
  "source_url": "https://maldita.es/..."
}
```

> El contenido original (imagen o texto) **nunca** se persiste — sólo metadatos y el hash del usuario.

### Qdrant — Base de conocimiento vectorial

Colección `fact_checks`. Cada punto almacena:
- Vector denso (BAAI/bge-m3, 1024 dimensiones) para búsqueda semántica.
- Vector disperso BM25 para búsqueda por keywords exactas.
- Payload: `title`, `content_summary`, `source_publisher`, `publish_date`, `url`.

Actualizada periódicamente por el servicio ETL.

### Ollama — Inferencia LLM local

Ejecuta modelos open-source dentro del clúster Docker. Opciones recomendadas según hardware:

| Modelo | RAM mínima | Uso |
|---|---|---|
| `llama3.2:3b` | 4 GB | Calidad alta, recomendado para producción |
| `qwen2.5:1.5b` | 2 GB | Más ligero, para servidores con poca RAM |

---

## Puesta en marcha

### Requisitos
- Docker + Docker Compose v2
- 8 GB RAM mínimo (16 GB recomendado)

### 1. Configurar entorno

```bash
cp .env.example .env
# Editar .env con TELEGRAM_BOT_TOKEN, API keys, etc.
```

### 2. Levantar infraestructura

```bash
# Solo bases de datos y brokers (para desarrollo)
docker compose up -d rabbitmq mongodb qdrant ollama

# Todo el sistema completo
docker compose up -d
```

### 3. Descargar el modelo LLM (primera vez)

```bash
docker exec -it verum_ollama ollama pull llama3.2:3b
```

### 4. Poblar la base de conocimiento

El servicio `etl_scheduler` arranca automáticamente con `docker compose up -d` y ejecuta el pipeline en el horario configurado en `ETL_SCHEDULE`. Para una carga inicial inmediata:

```bash
docker compose --profile etl run etl
```

### 5. Entrenar y exportar el modelo de visión

```bash
# En local con GPU
cd models/vision
python train.py --epochs 30 --batch-size 32
python evaluate.py --checkpoint weights/verum_cnn.pt --export-onnx
# Copia el .onnx a models/vision/weights/ — el worker lo monta como volumen read-only
```

### 6. Registrar el webhook de Telegram

Para desarrollo local con ngrok el proceso es completamente automático:

```bash
# Terminal 1: inicia ngrok
ngrok http 8000

# Terminal 2: registra el webhook (detecta la URL de ngrok automáticamente)
make webhook
```

El comando `make webhook`:
1. Lee la URL pública activa de ngrok (via su API local en :4040)
2. Registra el webhook en Telegram con tu `TELEGRAM_BOT_TOKEN`
3. Actualiza `WEBHOOK_BASE_URL` en el `.env`
4. Reinicia el gateway para aplicar cambios

Para ver el estado del webhook:
```bash
make webhook-info
```

Para eliminarlo:
```bash
make webhook-delete
```

Si despliegas en producción con dominio fijo (sin ngrok), edita `WEBHOOK_BASE_URL` en
el `.env` manualmente y reinicia el gateway. El lifespan handler en `main.py` se
encarga del registro automático al arrancar.

### Ver logs en tiempo real

```bash
docker compose logs -f worker_vision
docker compose logs -f worker_nlp
docker compose logs -f gateway
docker compose logs -f etl_scheduler
```

### Métricas Prometheus

El `worker_nlp` expone métricas en `http://localhost:9101/metrics`. Para verificar:

```bash
curl http://localhost:9101/metrics | grep verum_nlp
```

---

## Variables de entorno

Ver `.env.example` para la lista completa. Las críticas:

| Variable | Descripción |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Token del bot (BotFather) |
| `TELEGRAM_WEBHOOK_SECRET` | Header secret para validar POSTs de Telegram |
| `VISION_MODEL_PATH` | Ruta al `.onnx` dentro del contenedor `/app/weights/verum_cnn.onnx` |
| `OLLAMA_MODEL` | Modelo a usar: `llama3.2:3b` o `qwen2.5:1.5b` |
| `NLP_CONFIDENCE_THRESHOLD` | Umbral (0–1) bajo el que se activa el fallback a la API de Google |
| `GOOGLE_FACT_CHECK_API_KEY` | API key de Google Fact Check Tools |
| `NLP_MIN_TEXT_LENGTH` | Mensajes más cortos que este valor (chars) se ignoran |
