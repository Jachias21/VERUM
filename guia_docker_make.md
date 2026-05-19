# VERUM — Guía de Comandos Docker (`make`)

> Esta guía cubre todos los comandos del `Makefile` del proyecto VERUM.  
> El Makefile resuelve el problema de la caché de Docker asegurando que los builds
> siempre reconstruyan las imágenes desde cero cuando se lo pedimos explícitamente.

---

## Tabla de contenidos

1. [Requisitos](#1-requisitos)
2. [Configuración inicial](#2-configuración-inicial)
3. [Referencia rápida](#3-referencia-rápida)
4. [Ciclo de vida principal](#4-ciclo-de-vida-principal)
5. [Builds selectivos](#5-builds-selectivos)
6. [Logs](#6-logs)
7. [Shell y depuración](#7-shell-y-depuración)
8. [Estado y salud](#8-estado-y-salud)
9. [Herramientas NLP / RAG](#9-herramientas-nlp--rag)
10. [Limpieza](#10-limpieza)
11. [Flujos de trabajo habituales](#11-flujos-de-trabajo-habituales)

---

## 1. Requisitos

- [Docker](https://docs.docker.com/get-docker/) ≥ 24
- [Docker Compose](https://docs.docker.com/compose/) V2 (el que usa `docker compose`, sin guión)
- `make` instalado (`sudo apt install make` en Ubuntu / ya incluido en macOS)
- `curl` y `python3` para los comandos de health check y diagnóstico

---

## 2. Configuración inicial

Coloca el `Makefile` en la raíz del repositorio, al mismo nivel que `docker-compose.yml`.

```
verum/
├── docker-compose.yml
├── Makefile              ← aquí
├── services/
│   ├── gateway/
│   ├── worker_nlp/
│   ├── worker_vision/
│   └── etl/
└── ...
```


```makefile
BOT_SERVICE      := gateway
NLP_SERVICE      := worker_nlp
VISION_SERVICE   := worker_vision
RABBITMQ_SERVICE := rabbitmq
QDRANT_SERVICE   := qdrant
MONGO_SERVICE    := mongodb
OLLAMA_SERVICE   := ollama
ETL_SERVICE      := etl
```

Para ver la ayuda en cualquier momento:

```bash
make
# o también:
make help
```

---

## 3. Referencia rápida

| Comando | Qué hace |
|---|---|
| `make build` | Reconstruye **todas** las imágenes sin caché |
| `make up` | Levanta todos los servicios |
| `make down` | Para y elimina los contenedores (datos intactos) |
| `make rebuild` | **down + build + up** — el comando principal tras cambios |
| `make restart SERVICE=<nombre>` | Reinicia un servicio concreto |
| `make up-nlp` | Reconstruye y sube solo el worker NLP |
| `make up-bot` | Reconstruye y sube solo el gateway FastAPI |
| `make up-vision` | Reconstruye y sube solo el worker_vision |
| `make logs` | Logs de todos los servicios en tiempo real |
| `make logs-nlp` | Logs del worker NLP |
| `make logs-bot` | Logs del gateway FastAPI |
| `make shell-nlp` | Shell dentro del contenedor NLP |
| `make shell SERVICE=<nombre>` | Shell en cualquier contenedor |
| `make ps` | Lista los contenedores del proyecto |
| `make health` | Comprueba que los servicios responden |
| `make status` | Estado con puertos expuestos |
| `make nlp-test` | Ejecuta el test manual del pipeline NLP |
| `make etl-run` | Lanza el ETL para actualizar Qdrant |
| `make qdrant-check` | Consulta las colecciones de Qdrant |
| `make rabbit-check` | Estado de las colas de RabbitMQ |
| `make mongo-check` | Estadísticas de MongoDB |
| `make clean` | Para + elimina imágenes (datos intactos) |
| `make purge` | ⚠ Elimina **todo**, incluyendo volúmenes y datos |

---

## 4. Ciclo de vida principal

### `make build`

Reconstruye **todas** las imágenes del proyecto ignorando completamente la caché de
Docker. Usa `--no-cache` y `--pull` para asegurar que las capas base también se
actualizan.

```bash
make build
```

> Úsalo cuando cambies un `Dockerfile` o actualices dependencias en
> `requirements.txt`. Para cambios de código puro es más rápido usar
> [`make up-nlp`](#5-builds-selectivos).

---

### `make up`

Levanta todos los servicios en segundo plano con las imágenes que ya existen
localmente. **No reconstruye nada.**

```bash
make up
```

---

### `make down`

Para todos los contenedores y los elimina. Los volúmenes de datos (Qdrant, MongoDB)
se conservan.

```bash
make down
```

---

### `make rebuild`

Ejecuta en secuencia: `down → build → up`. Es el comando más seguro y completo
para aplicar cualquier cambio.

```bash
make rebuild
```

> **Cuándo usarlo:** siempre que no estés seguro de si el cambio es de código puro
> o afecta a dependencias / Dockerfiles. Tarda más pero garantiza que ningún
> contenedor usa código viejo.

---

### `make restart`

Reinicia un servicio concreto sin reconstruirlo. Útil para forzar una relectura de
variables de entorno o recuperarse de un crash.

```bash
# Reinicia el worker_nlp (valor por defecto)
make restart

# Reinicia el gateway
make restart SERVICE=gateway

# Reinicia RabbitMQ
make restart SERVICE=rabbitmq
```

---

## 5. Builds selectivos

Cuando solo modificas el código de un módulo, no hace falta reconstruir todo el
stack. Estos comandos reconstruyen y sustituyen **únicamente** el servicio indicado
sin afectar al resto (gracias a `--no-deps`).

### `make up-nlp`

Reconstruye el `worker_nlp` y lo sustituye en caliente.

```bash
make up-nlp
```

> **Cuándo usarlo:** después de cambiar lógica en el pipeline NLP, el módulo RAG,
> las llamadas a Qdrant, el prompt del LLM o cualquier archivo dentro de `services/worker_nlp/`.

### `make up-bot`

Reconstruye el gateway FastAPI.

```bash
make up-bot
```

> **Cuándo usarlo:** cuando modificas el webhook, el router de mensajes o la
> lógica de enrutamiento a las colas de RabbitMQ.

### `make up-vision`

Reconstruye el `worker_vision` (CNN).

```bash
make up-vision
```

> Comando principalmente para tu compañero, pero disponible en el mismo entorno.

### Builds solo (sin subir)

Si quieres construir la imagen sin levantar el contenedor todavía:

```bash
make build-nlp
make build-bot
make build-vision
```

---

## 6. Logs

Todos los comandos de logs son en tiempo real (`-f`) y muestran las últimas 100
líneas al arrancar.

### Todos los servicios

```bash
make logs
```

### Por servicio

```bash
make logs-nlp       # worker_nlp (SpaCy, LangChain, Qdrant client, Ollama)
make logs-bot       # gateway FastAPI / Telegram webhook
make logs-vision    # worker_vision (CNN, Grad-CAM)
make logs-rabbit    # RabbitMQ (colas topic_texts y topic_images)
make logs-qdrant    # Base de datos vectorial
make logs-mongo     # Base de datos analítica
make logs-ollama    # Servidor LLM local (Llama / Qwen)
make logs-etl       # Scheduler ETL (cron)
make logs-dashboard # Dashboard Streamlit
```

> **Tip NLP:** mientras pruebas con Telegram, ten `make logs-nlp` abierto en una
> terminal separada. Verás en tiempo real el NER, la búsqueda híbrida en Qdrant y
> la respuesta del LLM.

---

## 7. Shell y depuración

Abre una terminal `bash` dentro del contenedor en ejecución. Útil para inspeccionar
el estado interno, probar imports de Python o ejecutar scripts de forma manual.

### Contenedor worker_nlp

```bash
make shell-nlp
```

Una vez dentro puedes, por ejemplo:

```bash
# Comprobar que SpaCy carga el modelo correctamente
python -c "import spacy; nlp = spacy.load('es_core_news_lg'); print('OK')"

# Verificar la conexión con Qdrant
python -c "from qdrant_client import QdrantClient; c = QdrantClient(host='qdrant', port=6333); print(c.get_collections())"

# Probar el cliente de Ollama
curl http://ollama:11434/api/tags
```

### Contenedor gateway

```bash
make shell-bot
```

### Cualquier otro contenedor

```bash
make shell SERVICE=qdrant
make shell SERVICE=mongodb
make shell SERVICE=rabbitmq
make shell SERVICE=ollama
```

---

## 8. Estado y salud

### `make ps`

Lista los contenedores del proyecto con su estado actual.

```bash
make ps
```

Salida esperada cuando todo está bien:

```
NAME                    STATUS          PORTS
verum_gateway           Up 2 minutes    0.0.0.0:8000->8000/tcp
verum_worker_nlp        Up 2 minutes    0.0.0.0:9101->9101/tcp
verum_worker_vision     Up 2 minutes
verum_rabbitmq          Up 2 minutes    5672/tcp, 0.0.0.0:15672->15672/tcp
verum_qdrant            Up 2 minutes    0.0.0.0:6333->6333/tcp
verum_mongodb           Up 2 minutes    27017/tcp
verum_ollama            Up 2 minutes    0.0.0.0:11434->11434/tcp
verum_dashboard         Up 2 minutes    0.0.0.0:8501->8501/tcp
verum_etl_scheduler     Up 2 minutes
```

### `make status`

Igual que `ps` pero filtrado por proyecto y con formato de tabla más limpio.

```bash
make status
```

### `make health`

Llama a los endpoints de salud de cada servicio y muestra un OK / NO RESPONDE por
cada uno.

```bash
make health
```

Salida esperada:

```
── Health check VERUM ─────────────────────────────────────
  RabbitMQ Management UI  → OK
  Qdrant REST API         → OK
  Ollama API              → OK
  FastAPI / Gateway       → OK
  MongoDB                 → OK
```

> Si alguno muestra `NO RESPONDE`, usa `make logs-<servicio>` para ver qué está pasando.

---

## 9. Herramientas NLP / RAG

### `make nlp-test`

Ejecuta `python -m app.worker --test` dentro del contenedor `worker_nlp`. Sirve
para verificar manualmente que el pipeline completo funciona (NER → búsqueda híbrida
en Qdrant → síntesis con Ollama) sin necesidad de enviar un mensaje por Telegram.

```bash
make nlp-test
```

> Adapta el módulo `app.worker` para aceptar el flag `--test` si quieres
> un modo de prueba sin consumir la cola de RabbitMQ.

### `make etl-run`

Lanza el servicio `etl` como trabajo puntual (`docker compose run --rm etl`),
que ejecuta `python -m app.pipeline`: extrae artículos de
fact-checkers, genera embeddings y los indexa en Qdrant (upsert).

```bash
make etl-run
```

> Úsalo cuando quieras actualizar la knowledge base con nuevos desmentidos sin
> reiniciar todo el stack. El contenedor `etl_scheduler` repite este proceso
> automáticamente según el cron configurado en `ETL_SCHEDULE`.

### `make qdrant-check`

Consulta la API REST de Qdrant y lista las colecciones existentes.

```bash
make qdrant-check
```

Salida de ejemplo:

```json
{
    "result": {
        "collections": [
            { "name": "fact_checks" }
        ]
    },
    "status": "ok"
}
```

### `make rabbit-check`

Consulta la API de gestión de RabbitMQ y muestra el estado de las colas.

```bash
make rabbit-check
```

Salida de ejemplo:

```
  topic_texts:  0 msgs, 1 consumers
  topic_images: 0 msgs, 1 consumers
```

> Un consumer en `topic_texts` confirma que el `worker_nlp` está escuchando.
> Si aparece `0 consumers`, el worker no está conectado a la cola.

> **Nota:** el comando usa las credenciales de RabbitMQ definidas en `.env`
> (`RABBITMQ_USER` / `RABBITMQ_PASS`, por defecto `verum:verum_pass`).

### `make mongo-check`

Muestra cuántos documentos hay en la colección `queries` de MongoDB.

```bash
make mongo-check
```

---

## 10. Limpieza

### `make clean`

Para los contenedores y elimina las imágenes locales del proyecto. Los volúmenes
(datos de Qdrant y MongoDB) se conservan.

```bash
make clean
```

> Úsalo cuando quieras liberar espacio en disco sin perder el conocimiento indexado
> en Qdrant ni el histórico de MongoDB.

---

### `make purge`  ⚠

**Destructivo.** Elimina contenedores, imágenes **y volúmenes**. Qdrant y MongoDB
perderán todos sus datos. El comando pide confirmación antes de ejecutarse.

```bash
make purge
# → ¿Estás seguro? Escribe 'si' para continuar:
```

> Úsalo solo si necesitas un entorno completamente limpio, por ejemplo para
> reindexar la knowledge base de Qdrant desde cero o para diagnosticar un problema
> que podría venir de datos corruptos en los volúmenes.

---

## 11. Flujos de trabajo habituales

### Arranque inicial del proyecto

```bash
make build    # construye todas las imágenes por primera vez
make up       # levanta el stack
make health   # verifica que todo responde
make etl-run  # indexa la knowledge base en Qdrant
```

---

### Cambié código en el módulo NLP

```bash
make up-nlp         # reconstruye y sustituye solo el worker NLP
make logs-nlp       # verifica que arranca sin errores
make rabbit-check   # confirma que el worker está consumiendo topic_texts
```

---

### Cambié código en el gateway / webhook

```bash
make up-bot
make logs-bot
make health   # verifica que FastAPI responde en :8000
```

---

### Cambié un Dockerfile o requirements.txt

```bash
make rebuild  # down + build --no-cache + up (el más seguro)
make health
```

---

### Quiero ver qué pasa cuando llega un mensaje de Telegram

```bash
# Terminal 1
make logs-nlp

# Terminal 2 (opcional, para ver el gateway)
make logs-bot

# Envía el mensaje desde Telegram y observa los logs
```

---

### Necesito depurar el pipeline NLP manualmente

```bash
make shell-nlp
# dentro del contenedor:
python -m app.worker --test
```

---

### El sistema no responde, diagnóstico rápido

```bash
make ps           # ¿están los contenedores en estado "Up"?
make health       # ¿responden los endpoints?
make logs         # ¿hay errores recientes en algún servicio?
make rabbit-check # ¿las colas tienen consumidores?
```

---

### Quiero empezar completamente desde cero

```bash
make purge    # ⚠ borra todo, incluyendo datos
make build
make up
make etl-run
```