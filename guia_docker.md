
# 🛠️ Guía de Comandos Docker - Proyecto VERUM

Esta guía resume los comandos que hemos utilizado para gestionar los microservicios del TFM.

## 🚀 Comandos de Gestión Básica

| Comando | Descripción |
| :--- | :--- |
| `docker compose up -d` | Levanta todos los servicios en segundo plano. |
| `docker compose up -d --build` | Reconstruye las imágenes y levanta los servicios (úsalo tras cambiar un Dockerfile o requirements.txt). |
| `docker compose down` | Detiene y elimina todos los contenedores. |
| `docker compose ps` | Muestra el estado de los contenedores actuales. |

## 📦 Gestión de Servicios Específicos

Cuando solo cambias el código de un worker, no hace falta reiniciar todo:

| Comando | Descripción |
| :--- | :--- |
| `docker compose up -d --build worker_nlp` | Reconstruye y reinicia solo el Worker de NLP. |
| `docker compose up -d --build gateway` | Reconstruye y reinicia solo el API Gateway. |
| `docker compose restart ollama` | Reinicia el servicio de IA local (útil si se queda colgado). |

## 📝 Inspección de Logs (Vital para Debugging)

| Comando | Descripción |
| :--- | :--- |
| `docker compose logs -f worker_nlp` | Ver en tiempo real qué está haciendo el cerebro del bot. |
| `docker compose logs -f gateway` | Ver las peticiones que entran por Swagger o Telegram. |
| `docker compose logs --tail 100 worker_nlp` | Ver las últimas 100 líneas de log del worker. |

## 🧹 Mantenimiento y Limpieza

| Comando | Descripción |
| :--- | :--- |
| `docker builder prune -a -f` | **Comando de emergencia.** Borra la caché de construcción si Docker da errores raros de 'snapshot'. |
| `docker compose pull qdrant` | Descarga la última versión de la imagen de la base de datos. |
| `docker volume prune` | Borra todos los datos de las bases de datos (cuidado: borra MongoDB y Qdrant). |

## ⚡ Comandos Especiales del Proyecto

| Comando | Descripción |
| :--- | :--- |
| `docker compose --profile etl run etl` | Ejecuta el proceso de descarga de noticias desde los periódicos. |

---
*Nota: Si estás en Windows PowerShell, recuerda que algunos comandos de red como `curl` pueden fallar, por lo que recomendamos usar la interfaz de Swagger en `http://localhost:8000/docs`.*

