# Evaluación cuantitativa del módulo NLP — VERUM

## Descripción

Este documento explica el proceso de evaluación del módulo RAG del sistema VERUM,
incluyendo la construcción del gold set, la ejecución del script y la interpretación
de los resultados. El objetivo es proporcionar la métrica cuantitativa equivalente
al 85% de accuracy del módulo de visión, aplicada al módulo NLP.

---

## Gold set (`tests/golden/gold_nlp.jsonl`)

### Estructura de cada ejemplo

```json
{"id": "fake_001", "text": "...", "expected_verdict": "FAKE", "category": "phishing"}
```

| Campo              | Descripción                                              |
|--------------------|----------------------------------------------------------|
| `id`               | Identificador único del ejemplo                          |
| `text`             | Texto a verificar (≥ 50 chars salvo gibberish)           |
| `expected_verdict` | Etiqueta manual: `FAKE`, `REAL` o `UNVERIFIED`           |
| `category`         | Subcategoría temática del ejemplo                        |

### Distribución de clases (60 ejemplos)

| Clase        | N  | Categorías                                              |
|--------------|----|---------------------------------------------------------|
| `FAKE`       | 30 | `cybersecurity` (10), `socio-political` (10), `phishing` (10) |
| `REAL`       | 15 | `history` (6), `science` (6), `politics` (2), `economics` (1) |
| `UNVERIFIED` | 15 | `off-topic` (5), `gibberish` (5), `ambiguous` (5)      |

### Criterios de selección

- **FAKE**: Se seleccionaron bulos documentados y circulantes en España, divididos
  en tres grupos de alta frecuencia en redes sociales: amenazas de ciberseguridad
  (números peligrosos, virus en audio, WhatsApp de pago), desinformación política
  y electoral, y campañas de phishing con marcas conocidas (Mercadona, Correos, DHL).

- **REAL**: Hechos verificables y documentados con fecha y fuente oficial: tratados
  internacionales, decisiones institucionales, datos científicos publicados. El sistema
  debería recuperar artículos de fact-checking o contexto relevante en Qdrant.

- **UNVERIFIED**: Tres subcategorías de diferente naturaleza:
  - *off-topic*: noticias inventadas pero plausibles, sin evidencia indexada.
  - *gibberish*: texto sin sentido que activa `is_gibberish()`.
  - *ambiguous*: mensajes vagos sin entidades verificables que deberían degradar a UNVERIFIED.

### Limitaciones del gold set

1. **Asimetría del sistema**: VERUM está diseñado principalmente para detectar bulos
   (FAKE), por lo que su recall en la clase REAL puede ser subóptimo si Qdrant no
   tiene artículos de verificación positiva indexados.

2. **Dependencia de Qdrant**: Los ejemplos FAKE de phishing dependen de que Maldita.es
   o Newtral tengan los desmentidos indexados. Si la base de datos local está vacía,
   el sistema escalará a Google FC / GNews con resultados más variables.

3. **Evolución temporal**: Los bulos y hechos son válidos en el momento de creación
   del gold set (mayo 2026). Algunos pueden volverse irrelevantes o cambiar de estado
   con el tiempo.

4. **Tamaño del gold set**: 60 ejemplos es suficiente para una primera evaluación
   académica pero insuficiente para intervalos de confianza robustos. Se recomienda
   ampliar a 200+ ejemplos en futuras iteraciones.

---

## Ejecución

### Con Docker (recomendado)

```bash
# Paso 1: asegurarse de que todos los servicios estén corriendo
make up

# Paso 2: esperar a que Qdrant y Ollama estén listos
make health

# Paso 3: ejecutar la evaluación
make eval-nlp
```

Los informes se generan en `reports/` dentro del contenedor y son montados
como volumen en el directorio local `reports/` del host.

### Sin Docker (local, para desarrollo)

```bash
# Desde la raíz del repositorio, con el entorno virtual activado:
python scripts/eval_nlp.py \
  --gold tests/golden/gold_nlp.jsonl \
  --out reports/

# Para probar rápido con solo los primeros 10 ejemplos:
python scripts/eval_nlp.py --gold tests/golden/gold_nlp.jsonl --out reports/ --limit 10
```

Requisitos locales adicionales: `scikit-learn`, `matplotlib`, `numpy`.

---

## Outputs

### `reports/eval_summary.json`

Métricas en formato máquina-legible. Estructura principal:

```json
{
  "total_examples": 60,
  "errors": 0,
  "macro_f1": 0.812,
  "per_class": {
    "FAKE":       {"precision": 0.91, "recall": 0.87, "f1": 0.89, "support": 30},
    "REAL":       {"precision": 0.80, "recall": 0.73, "f1": 0.76, "support": 15},
    "UNVERIFIED": {"precision": 0.75, "recall": 0.80, "f1": 0.77, "support": 15}
  },
  "confusion_matrix": [[26, 1, 3], [1, 11, 3], [2, 1, 12]],
  "confusion_matrix_labels": ["FAKE", "REAL", "UNVERIFIED"],
  "latency_mean_ms": 3420,
  "latency_p50_ms": 2800,
  "latency_p95_ms": 8100,
  "url_coverage_pct": 62.5,
  "accuracy_by_category": {
    "phishing": 0.9, "cybersecurity": 0.8, "gibberish": 1.0, ...
  }
}
```

### `reports/eval_report.html`

Informe visual con:

1. **Resumen**: totales, latencia, cobertura de URL.
2. **Tabla de métricas por clase**: precisión, recall, F1 y soporte.
3. **Matriz de confusión**: imagen PNG embebida en base64, generada con matplotlib.
4. **Accuracy por categoría**: permite identificar qué subcategorías funcionan mejor/peor.
5. **Listado de fallos**: id, clase esperada, clase predicha y error (si lo hubo).

---

## Interpretación de resultados

### Umbral de éxito del TFM

El criterio de éxito definido en el TFM es **macro-F1 ≥ 0.75**. El informe HTML
muestra en verde si se supera este umbral, en rojo si no.

### ¿Por qué macro-F1 y no accuracy?

Las clases están desbalanceadas (30/15/15). La accuracy global puede ser alta
aunque el sistema falle sistemáticamente en REAL o UNVERIFIED. El macro-F1
promedia el F1 de cada clase con igual peso, penalizando el desbalance.

### Señales de alerta

| Síntoma                                  | Causa probable                                   |
|------------------------------------------|--------------------------------------------------|
| F1(FAKE) < 0.70                          | Qdrant sin desmentidos indexados                 |
| F1(REAL) < 0.50                          | LLM sesgo hacia FAKE; pocos artículos positivos  |
| Accuracy gibberish < 1.0                 | Umbral `is_gibberish()` demasiado permisivo       |
| p95 > 30s                                | Ollama sobrecargado o modelo demasiado grande    |
| `errors` > 0                             | Ver columna "Error" en el informe HTML           |

---

## Archivos versionados vs ignorados

| Archivo                          | Estado       | Razón                                    |
|----------------------------------|--------------|------------------------------------------|
| `tests/golden/gold_nlp.jsonl`    | Versionado   | Reproducibilidad del benchmark           |
| `scripts/eval_nlp.py`            | Versionado   | Código de evaluación                     |
| `scripts/README_eval.md`         | Versionado   | Documentación                            |
| `reports/eval_summary.json`      | **Ignorado** | Output variable según entorno            |
| `reports/eval_report.html`       | **Ignorado** | Output variable según entorno            |
