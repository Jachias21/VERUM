# =============================================================================
#  VERUM - Makefile de gestion Docker
#  Uso: make <target> [SERVICE=<nombre>]
#  Ejemplo: make logs-nlp | make restart SERVICE=bot | make shell SERVICE=qdrant
# =============================================================================

# -- Configuracion -----------------------------------------------------------
COMPOSE_FILE      := docker-compose.yml
PROJECT_NAME      := verum

# Nombres de servicio tal como aparecen en docker-compose.yml
BOT_SERVICE       := gateway
NLP_SERVICE       := worker_nlp
VISION_SERVICE    := worker_vision
RABBITMQ_SERVICE  := rabbitmq
QDRANT_SERVICE    := qdrant
MONGO_SERVICE     := mongodb
OLLAMA_SERVICE    := ollama
ETL_SERVICE       := etl
DASHBOARD_SERVICE := dashboard
TEST_SERVICE      := test_runner

# Auto-detect python command
PYTHON := $(shell python3 --version >/dev/null 2>&1 && echo python3 || echo python)

# -- Default -----------------------------------------------------------------
.DEFAULT_GOAL := help

# Marca todos los targets como "phony" (no son archivos)
.PHONY: help build up down restart rebuild clean purge \
        logs logs-nlp logs-bot logs-vision logs-rabbit logs-qdrant logs-mongo logs-ollama logs-etl logs-dashboard \
        shell shell-nlp shell-bot \
        ps health status \
        qdrant-check mongo-check rabbit-check \
        ollama-pull ollama-list \
        nlp-test etl-run \
        test-build test test-nlp test-etl test-gateway test-integration \
        webhook webhook-info webhook-delete

# =============================================================================
#  AYUDA
# =============================================================================
help:
	@echo ""
	@echo "============================================"
	@echo "  VERUM -- Comandos disponibles"
	@echo "============================================"
	@echo ""
	@echo "--- Ciclo de vida ---------------------------"
	@echo "  make build          Reconstruye TODAS las imagenes (sin cache)"
	@echo "  make up             Levanta todos los servicios"
	@echo "  make down           Para y elimina contenedores (datos intactos)"
	@echo "  make rebuild        down + build + up (el mas seguro tras cambios)"
	@echo "  make restart        Reinicia un servicio: make restart SERVICE=worker_nlp"
	@echo ""
	@echo "--- Builds selectivos -----------------------"
	@echo "  make up-nlp         Reconstruye y sube solo worker_nlp"
	@echo "  make up-bot         Reconstruye y sube solo gateway"
	@echo "  make up-vision      Reconstruye y sube solo worker_vision"
	@echo ""
	@echo "--- Tests -----------------------------------"
	@echo "  make test           Ejecuta TODOS los tests (dentro de Docker)"
	@echo "  make test-nlp       Tests del worker NLP (NER, cache, RAG)"
	@echo "  make test-etl       Tests del pipeline ETL"
	@echo "  make test-gateway   Tests del gateway/router"
	@echo "  make test-integration  Tests de integracion end-to-end"
	@echo "  make test-build     Solo construye la imagen de tests (sin ejecutar)"
	@echo ""
	@echo "--- Logs ------------------------------------"
	@echo "  make logs           Todos los servicios (tiempo real)"
	@echo "  make logs-nlp       Worker NLP"
	@echo "  make logs-bot       Gateway FastAPI"
	@echo "  make logs-vision    Worker Vision"
	@echo "  make logs-etl       Scheduler ETL"
	@echo "  make logs-dashboard Dashboard Streamlit"
	@echo ""
	@echo "--- Estado y diagnostico --------------------"
	@echo "  make ps             Lista contenedores del proyecto"
	@echo "  make health         Health check de todos los servicios"
	@echo "  make status         Estado con puertos expuestos"
	@echo ""
	@echo "--- Herramientas NLP / RAG ------------------"
	@echo "  make nlp-test       Test manual del pipeline NLP"
	@echo "  make etl-run        Lanza ETL (actualiza knowledge base)"
	@echo "  make qdrant-check   Colecciones en Qdrant"
	@echo "  make rabbit-check   Estado de colas RabbitMQ"
	@echo "  make mongo-check    Documentos en MongoDB"
	@echo "  make ollama-pull    Descarga el modelo OLLAMA_MODEL en el contenedor"
	@echo "  make ollama-list    Lista los modelos instalados en Ollama"
	@echo ""
	@echo "--- Webhook Telegram (con ngrok) ------------"
	@echo "  make webhook        Detecta URL de ngrok y registra webhook automaticamente"
	@echo "  make webhook-info   Muestra el estado del webhook en Telegram"
	@echo "  make webhook-delete Elimina el webhook registrado"
	@echo ""
	@echo "--- Limpieza --------------------------------"
	@echo "  make clean          Elimina imagenes (datos de Qdrant/MongoDB intactos)"
	@echo "  make purge          Elimina TODO incluyendo volumenes y datos"
	@echo ""

# =============================================================================
#  CICLO DE VIDA PRINCIPAL
# =============================================================================

build:
	@echo ""
	@echo "============================================"
	@echo "  VERUM -- Reconstruyendo imagenes"
	@echo "============================================"
	@echo ""
	@echo "[>>] Ejecutando build --no-cache --pull ..."
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) build --no-cache --pull 2>&1 | tail -5
	@echo "[OK] Build completado."
	@echo ""

up:
	@echo ""
	@echo "[>>] Levantando todos los servicios..."
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) up -d 2>&1 | tail -5
	@echo "[OK] Servicios en marcha. Usa 'make ps' para verificar."
	@echo ""

down:
	@echo ""
	@echo "[>>] Parando y eliminando contenedores..."
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) down --remove-orphans 2>&1 | tail -5
	@echo "[OK] Contenedores eliminados."
	@echo ""

rebuild: down build up
	@echo "[OK] Rebuild completo finalizado."
	@echo ""

SERVICE ?= $(NLP_SERVICE)
restart:
	@echo ""
	@echo "[>>] Reiniciando servicio: $(SERVICE)..."
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) restart $(SERVICE) 2>&1 | tail -5
	@echo "[OK] $(SERVICE) reiniciado."
	@echo ""

# =============================================================================
#  BUILDS SELECTIVOS
# =============================================================================

build-nlp:
	@echo ""
	@echo "[>>] Reconstruyendo worker_nlp..."
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) build --no-cache $(NLP_SERVICE) 2>&1 | tail -5
	@echo "[OK] worker_nlp reconstruido."
	@echo ""

build-bot:
	@echo ""
	@echo "[>>] Reconstruyendo gateway..."
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) build --no-cache $(BOT_SERVICE) 2>&1 | tail -5
	@echo "[OK] gateway reconstruido."
	@echo ""

build-vision:
	@echo ""
	@echo "[>>] Reconstruyendo worker_vision..."
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) build --no-cache $(VISION_SERVICE) 2>&1 | tail -5
	@echo "[OK] worker_vision reconstruido."
	@echo ""

## Reconstruye y sube solo el worker_nlp (sin tocar el resto)
up-nlp: build-nlp
	@echo "[>>] Subiendo worker_nlp..."
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) up -d --no-deps $(NLP_SERVICE) 2>&1 | tail -5
	@echo "[OK] worker_nlp actualizado."
	@echo ""

## Reconstruye y sube solo el gateway FastAPI
up-bot: build-bot
	@echo "[>>] Subiendo gateway..."
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) up -d --no-deps $(BOT_SERVICE) 2>&1 | tail -5
	@echo "[OK] gateway actualizado."
	@echo ""

## Reconstruye y sube solo el worker_vision
up-vision: build-vision
	@echo "[>>] Subiendo worker_vision..."
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) up -d --no-deps $(VISION_SERVICE) 2>&1 | tail -5
	@echo "[OK] worker_vision actualizado."
	@echo ""

# =============================================================================
#  LOGS  (interactivos -- sin supresion de output)
# =============================================================================

## Logs de todos los servicios en tiempo real
logs:
	docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) logs -f --tail=100

logs-nlp:
	docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) logs -f --tail=100 $(NLP_SERVICE)

logs-bot:
	docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) logs -f --tail=100 $(BOT_SERVICE)

logs-vision:
	docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) logs -f --tail=100 $(VISION_SERVICE)

logs-rabbit:
	docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) logs -f --tail=100 $(RABBITMQ_SERVICE)

logs-qdrant:
	docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) logs -f --tail=100 $(QDRANT_SERVICE)

logs-mongo:
	docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) logs -f --tail=100 $(MONGO_SERVICE)

logs-ollama:
	docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) logs -f --tail=100 $(OLLAMA_SERVICE)

logs-etl:
	docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) logs -f --tail=100 etl_scheduler

logs-dashboard:
	docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) logs -f --tail=100 $(DASHBOARD_SERVICE)

# =============================================================================
#  SHELL / DEPURACION  (interactivos -- sin supresion de output)
# =============================================================================

## Shell dentro del contenedor worker_nlp (para depurar SpaCy, LangChain, Qdrant client...)
shell-nlp:
	docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) exec $(NLP_SERVICE) /bin/bash

## Shell dentro del contenedor gateway FastAPI
shell-bot:
	docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) exec $(BOT_SERVICE) /bin/bash

## Shell generico: make shell SERVICE=qdrant
shell:
	docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) exec $(SERVICE) /bin/bash

# =============================================================================
#  ESTADO Y SALUD
# =============================================================================

## Lista los contenedores del proyecto con su estado
ps:
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) ps

## Estado detallado (incluye puertos mapeados)
status:
	@echo ""
	@echo "--- Contenedores VERUM ---"
	@docker ps --filter "name=$(PROJECT_NAME)" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""

## Comprueba la salud de los servicios clave usando sus APIs de health
health:
	@echo ""
	@echo "--- Health Check VERUM -------------------------"
	@printf "  %-28s" "RabbitMQ"; \
	  curl -sf http://localhost:15672 > /dev/null 2>&1 && echo "[OK]" || echo "[!!] NO RESPONDE"
	@printf "  %-28s" "Qdrant"; \
	  curl -sf http://localhost:6333/healthz > /dev/null 2>&1 && echo "[OK]" || echo "[!!] NO RESPONDE"
	@printf "  %-28s" "Ollama"; \
	  curl -sf http://localhost:11434/api/tags > /dev/null 2>&1 && echo "[OK]" || echo "[!!] NO RESPONDE"
	@printf "  %-28s" "Gateway (FastAPI)"; \
	  curl -sf http://localhost:8000/health > /dev/null 2>&1 && echo "[OK]" || echo "[!!] NO RESPONDE"
	@printf "  %-28s" "MongoDB"; \
	  docker compose -p $(PROJECT_NAME) exec -T $(MONGO_SERVICE) \
	    mongosh --quiet --eval "db.adminCommand('ping').ok" 2>/dev/null | grep -q "1" \
	    && echo "[OK]" || echo "[!!] NO RESPONDE"
	@echo "------------------------------------------------"
	@echo ""

# =============================================================================
#  HERRAMIENTAS NLP / RAG
# =============================================================================

## Ejecuta un mensaje de prueba manualmente por el pipeline NLP
nlp-test:
	@echo ""
	@echo "[>>] Ejecutando test del pipeline NLP..."
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) exec $(NLP_SERVICE) \
	  python -m app.worker --test
	@echo "[OK] Test NLP completado."
	@echo ""

## Lanza el pipeline ETL para re-indexar articulos en Qdrant
etl-run:
	@echo ""
	@echo "[>>] Lanzando pipeline ETL (actualizacion de knowledge base)..."
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) --profile etl run --rm $(ETL_SERVICE) 2>&1 | tail -5
	@echo "[OK] ETL completado."
	@echo ""

## Consulta las colecciones existentes en Qdrant
qdrant-check:
	@echo ""
	@echo "--- Colecciones en Qdrant ---"
	@curl -s http://localhost:6333/collections | $(PYTHON) -m json.tool 2>/dev/null \
	  || echo "[!!] No se puede conectar a Qdrant en :6333"
	@echo ""

## Consulta el estado de las colas en RabbitMQ
rabbit-check:
	@echo ""
	@echo "--- Colas en RabbitMQ ---"
	@curl -sf -u $${RABBITMQ_USER:-verum}:$${RABBITMQ_PASS:-verum_pass} http://localhost:15672/api/queues | \
	  $(PYTHON) -c "import sys,json; qs=json.load(sys.stdin); \
	  [print(f'  {q[\"name\"]}: {q[\"messages\"]} msgs, {q[\"consumers\"]} consumers') for q in qs]" \
	  2>/dev/null || echo "[!!] No se puede conectar a RabbitMQ management en :15672"
	@echo ""

## Consulta estadisticas basicas de MongoDB
mongo-check:
	@echo ""
	@echo "--- Estadisticas MongoDB ---"
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) exec -T $(MONGO_SERVICE) \
	  mongosh --quiet --eval \
	  "db.getSiblingDB('verum').queries.countDocuments({}).then(n => print('  Total queries registradas: ' + n))" \
	  2>/dev/null || echo "[!!] No se puede conectar a MongoDB"
	@echo ""

## Descarga el modelo OLLAMA_MODEL en el contenedor Ollama
ollama-pull:
	@echo ""
	@echo "[>>] Descargando modelo $${OLLAMA_MODEL:-llama3.2:3b} en Ollama..."
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) exec $(OLLAMA_SERVICE) ollama pull $${OLLAMA_MODEL:-llama3.2:3b}
	@echo "[OK] Modelo descargado."
	@echo ""

## Lista los modelos instalados en el contenedor Ollama
ollama-list:
	@echo ""
	@echo "--- Modelos en Ollama ---"
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) exec $(OLLAMA_SERVICE) ollama list
	@echo ""

# =============================================================================
#  WEBHOOK TELEGRAM (con ngrok)
# =============================================================================

## Detecta la URL de ngrok y registra el webhook con Telegram automaticamente
webhook:
	@bash scripts/setup_webhook.sh
	@echo ""
	@echo "[>>] Reiniciando gateway para aplicar la nueva URL..."
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) restart $(BOT_SERVICE) > /dev/null 2>&1
	@echo "[OK] Gateway reiniciado."
	@echo ""

## Muestra el estado actual del webhook registrado en Telegram
webhook-info:
	@bash scripts/webhook_info.sh

## Elimina el webhook registrado en Telegram
webhook-delete:
	@bash scripts/webhook_delete.sh

# =============================================================================
#  LIMPIEZA
# =============================================================================

## Para contenedores y elimina las imagenes del proyecto (los volumenes se conservan)
clean: down
	@echo ""
	@echo "[>>] Eliminando imagenes del proyecto..."
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) down --rmi local 2>&1 | tail -3
	@echo "[OK] Imagenes eliminadas. Los datos de Qdrant/MongoDB se conservan."
	@echo ""

## [!!] DESTRUCTIVO: elimina contenedores, imagenes Y volumenes (se pierden los datos)
purge:
	@echo ""
	@echo "============================================"
	@echo "  [!!] ATENCION: accion destructiva"
	@echo "  Esto eliminara Qdrant, MongoDB, imagenes"
	@echo "  y todos los volumenes de datos."
	@echo "============================================"
	@echo ""
	@read -p "  Escribe 'si' para continuar: " confirm && \
	  [ "$$confirm" = "si" ] || (echo "" && echo "[!!] Cancelado." && echo "" && exit 1)
	@echo ""
	@echo "[>>] Eliminando contenedores, volumenes e imagenes..."
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) down --volumes --remove-orphans --rmi local 2>&1 | tail -3
	@docker volume prune -f --filter "label=com.docker.compose.project=$(PROJECT_NAME)" > /dev/null 2>&1
	@echo "[OK] Entorno completamente limpio."
	@echo ""

# =============================================================================
#  TESTS (se ejecutan DENTRO de Docker — las dependencias no se instalan local)
# =============================================================================

## Construye (o reconstruye) la imagen de tests sin ejecutarlos
test-build:
	@echo ""
	@echo "[>>] Construyendo imagen de tests..."
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) --profile test build $(TEST_SERVICE) 2>&1 | tail -5
	@echo "[OK] Imagen test_runner lista."
	@echo ""

## Ejecuta TODOS los tests dentro del contenedor
test: test-build
	@echo ""
	@echo "============================================"
	@echo "  VERUM -- Ejecutando todos los tests"
	@echo "============================================"
	@echo ""
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) --profile test run --rm $(TEST_SERVICE)
	@echo ""

## Ejecuta solo los tests del worker NLP (NER, cache, RAG)
test-nlp: test-build
	@echo ""
	@echo "[>>] Tests worker_nlp (NER, cache, RAG)..."
	@echo ""
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) --profile test run --rm $(TEST_SERVICE) \
	  pytest tests/test_worker_nlp/ -v --tb=short --color=yes
	@echo ""

## Ejecuta solo los tests del pipeline ETL
test-etl: test-build
	@echo ""
	@echo "[>>] Tests ETL pipeline..."
	@echo ""
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) --profile test run --rm $(TEST_SERVICE) \
	  pytest tests/test_etl/ -v --tb=short --color=yes
	@echo ""

## Ejecuta solo los tests del gateway/router
test-gateway: test-build
	@echo ""
	@echo "[>>] Tests gateway/router..."
	@echo ""
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) --profile test run --rm $(TEST_SERVICE) \
	  pytest tests/test_gateway/ -v --tb=short --color=yes
	@echo ""

## Ejecuta solo los tests de integracion end-to-end
test-integration: test-build
	@echo ""
	@echo "[>>] Tests de integracion end-to-end..."
	@echo ""
	@docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) --profile test run --rm $(TEST_SERVICE) \
	  pytest tests/test_integration/ -v --tb=short --color=yes
	@echo ""