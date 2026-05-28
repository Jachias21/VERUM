"""
generate_fake.py — genera imágenes sintéticas con la Hugging Face Inference API
(SDXL-base-1.0) para construir la clase FAKE del dataset propio de VERUM.

Uso:
    python scripts/build_dataset/generate_fake.py --hf-token <TOKEN>
    python scripts/build_dataset/generate_fake.py \\
        --output-dir data/custom/fake/ \\
        --images-per-query 50 \\
        --hf-token <TOKEN>

    # O con variable de entorno:
    HF_TOKEN=<TOKEN> python scripts/build_dataset/generate_fake.py

Dependencias:
    pip install huggingface_hub Pillow
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import uuid
from pathlib import Path

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

# ---------------------------------------------------------------------------
# Modelo & retry
# ---------------------------------------------------------------------------
_HF_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
_MAX_RETRIES = 6        # reintentos máximos por imagen
_BACKOFF_BASE = 2.0     # base del backoff exponencial (segundos)
_BACKOFF_MAX = 120.0    # techo del backoff

# ---------------------------------------------------------------------------
# Prompts de generación — mismas categorías que scrape_real.py pero
# formulados como instrucciones de imagen fotorrealista para SDXL.
# Sufijo común que mejora la calidad y el fotorrealismo del modelo.
# ---------------------------------------------------------------------------
_SUFFIX = ", photorealistic, high quality, detailed, 4k, sharp focus, natural lighting"

PROMPTS: dict[str, list[str]] = {
    "personas_famosas": [
        f"a photo of a famous politician giving a speech at a press conference{_SUFFIX}",
        f"a celebrity walking the red carpet at an awards ceremony{_SUFFIX}",
        f"world leaders shaking hands at an official summit{_SUFFIX}",
        f"a famous musician performing live on stage in a concert{_SUFFIX}",
        f"an athlete competing in a professional sports event{_SUFFIX}",
    ],
    "eventos_sociales": [
        f"a large protest demonstration with crowds holding signs in a city street{_SUFFIX}",
        f"aerial view of a flooded city after a natural disaster{_SUFFIX}",
        f"a political rally with thousands of people gathered outdoors{_SUFFIX}",
        f"a colorful street festival celebration with people dancing{_SUFFIX}",
        f"a humanitarian aid distribution in a refugee camp{_SUFFIX}",
    ],
    "paisajes_lugares": [
        f"a dramatic mountain landscape with snow peaks and pine forest{_SUFFIX}",
        f"a busy urban cityscape at night with skyscrapers and neon lights{_SUFFIX}",
        f"a tropical ocean beach at golden sunset{_SUFFIX}",
        f"a lush forest with a waterfall and a river{_SUFFIX}",
        f"an aerial view of sand dunes in the Sahara desert{_SUFFIX}",
    ],
    "animales": [
        f"a lion hunting in the savannah, wildlife photography{_SUFFIX}",
        f"a golden retriever and a tabby cat playing together indoors{_SUFFIX}",
        f"a flock of birds in flight over a lake at dawn{_SUFFIX}",
        f"colorful tropical fish swimming in a coral reef{_SUFFIX}",
        f"cows and sheep grazing on a green countryside farm{_SUFFIX}",
    ],
    "comida": [
        f"an elegantly plated gourmet meal in a fine dining restaurant{_SUFFIX}",
        f"a vibrant farmers market stall full of fresh fruit and vegetables{_SUFFIX}",
        f"a street food vendor cooking in a busy Asian night market{_SUFFIX}",
        f"freshly baked artisan bread and pastries in a bakery{_SUFFIX}",
        f"a traditional Spanish paella dish served in a pan{_SUFFIX}",
    ],
    "objetos_cotidianos": [
        f"a tidy home office desk with laptop, books and a coffee mug{_SUFFIX}",
        f"a workbench full of hand tools and hardware in a workshop{_SUFFIX}",
        f"a flat lay of modern electronics gadgets and smartphones{_SUFFIX}",
        f"a collection of fashionable clothing and accessories on a rack{_SUFFIX}",
        f"an open book with stationery items on a wooden desk{_SUFFIX}",
    ],
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter, capped at _BACKOFF_MAX seconds."""
    delay = min(_BACKOFF_BASE ** attempt, _BACKOFF_MAX)
    return delay


def _is_valid_image(image: "PIL.Image.Image", min_px: int = 256) -> bool:  # type: ignore[name-defined]
    """Return True only if the PIL image has both sides ≥ min_px."""
    try:
        w, h = image.size
        return w >= min_px and h >= min_px
    except Exception:  # noqa: BLE001
        return False


def _generate_one(
    prompt: str,
    client: InferenceClient,
) -> "PIL.Image.Image | None":  # type: ignore[name-defined]
    """
    Llama a InferenceClient.text_to_image y devuelve un PIL Image, o None si
    todos los reintentos fallan.

    Maneja:
    - 503 (modelo cargando) — backoff exponencial
    - 429 (rate-limit)      — backoff exponencial
    - Errores de red        — backoff exponencial
    - 4xx permanentes       — fallo inmediato (sin reintentos)
    """
    for attempt in range(_MAX_RETRIES):
        try:
            image = client.text_to_image(
                prompt=prompt,
                model=_HF_MODEL,
            )
            return image  # PIL.Image.Image

        except HfHubHTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0

            if status == 503:
                # Modelo cargando (cold start) — backoff generoso
                wait = _backoff(attempt + 2)  # empieza en 4s, sube a 120s
                log.warning(
                    "  ⏳ Modelo cargando (503). Esperando %.0fs (intento %d/%d)…",
                    wait, attempt + 1, _MAX_RETRIES,
                )
                time.sleep(wait)

            elif status == 429:
                delay = _backoff(attempt + 1)
                log.warning(
                    "  ⚠️  Rate limit (429). Backoff %.0fs (intento %d/%d)…",
                    delay, attempt + 1, _MAX_RETRIES,
                )
                time.sleep(delay)

            elif 400 <= status < 500:
                # Error de cliente no recuperable (ej. 401, 403)
                log.error("  ✗ HTTP %d (no recuperable): %s", status, str(exc)[:200])
                return None

            else:
                # 5xx u otros — backoff y reintento
                delay = _backoff(attempt + 1)
                log.warning(
                    "  ✗ HTTP %d. Reintento %d/%d en %.0fs…",
                    status, attempt + 1, _MAX_RETRIES, delay,
                )
                time.sleep(delay)

        except Exception as exc:  # noqa: BLE001  (timeout, network error, etc.)
            delay = _backoff(attempt + 1)
            log.warning(
                "  ✗ Error (%s). Reintento %d/%d en %.0fs…",
                type(exc).__name__, attempt + 1, _MAX_RETRIES, delay,
            )
            time.sleep(delay)

    log.error("  ✗ Fallaron los %d reintentos para este prompt.", _MAX_RETRIES)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera imágenes sintéticas con SDXL (HF API) para el dataset VERUM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/custom/fake"),
        help="Directorio raíz donde se guardan las imágenes generadas.",
    )
    parser.add_argument(
        "--images-per-query",
        type=int,
        default=50,
        help="Número de imágenes a generar por cada prompt.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face API token. Alternativa: variable de entorno HF_TOKEN.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=256,
        help="Tamaño mínimo (px) en cada dimensión para conservar la imagen.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Pausa mínima en segundos entre peticiones a la API.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Token ────────────────────────────────────────────────────────────────
    hf_token: str | None = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE")
    if not hf_token:
        log.error(
            "No se encontró el token de Hugging Face.\n"
            "  Pásalo con --hf-token <TOKEN>  o  exporta HF_TOKEN=<TOKEN>."
        )
        sys.exit(1)

    output_root: Path = args.output_dir
    images_per_query: int = args.images_per_query
    min_size: int = args.min_size
    delay: float = args.delay

    output_root.mkdir(parents=True, exist_ok=True)

    client = InferenceClient(token=hf_token)

    # summary: category → kept images count
    summary: dict[str, int] = {}

    total_categories = len(PROMPTS)

    for cat_idx, (category, prompts) in enumerate(PROMPTS.items(), start=1):
            cat_dir = output_root / category
            cat_dir.mkdir(parents=True, exist_ok=True)

            log.info(
                "━━━ Categoría %d/%d: %s (%d prompts × %d imágenes) ━━━",
                cat_idx, total_categories, category, len(prompts), images_per_query,
            )

            cat_kept = 0

            for p_idx, prompt in enumerate(prompts, start=1):
                log.info(
                    "  [%d/%d] Prompt: «%s»",
                    p_idx, len(prompts), prompt[:80],
                )

                prompt_kept = 0
                prompt_failed = 0
                target = images_per_query

                for img_num in range(1, target + 1):
                    # Progress indicator every 10 images
                    if img_num % 10 == 1 and img_num > 1:
                        log.info(
                            "         … %d/%d generadas (✓%d  ✗%d)",
                            img_num - 1, target, prompt_kept, prompt_failed,
                        )

                    image = _generate_one(prompt, client)

                    if image is None or not _is_valid_image(image, min_px=min_size):
                        if image is not None:
                            log.debug("  Imagen descartada (tamaño < %dpx).", min_size)
                        prompt_failed += 1
                        continue

                    # Save with unique filename
                    filename = cat_dir / f"{uuid.uuid4().hex}.png"
                    image.save(str(filename))
                    prompt_kept += 1

                    # Polite delay between requests
                    if img_num < target:
                        time.sleep(delay)

                log.info(
                    "         ✓ %d imágenes guardadas  |  %d fallidas/descartadas",
                    prompt_kept, prompt_failed,
                )
                cat_kept += prompt_kept

            summary[category] = cat_kept
            log.info("  → %s total: %d imágenes.", category, cat_kept)

    # ── Resumen final ────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  RESUMEN — imágenes generadas por categoría")
    print("═" * 60)
    grand_total = 0
    for category, count in summary.items():
        print(f"  {category:<30s}  {count:>5d} imágenes")
        grand_total += count
    print("─" * 60)
    print(f"  {'TOTAL':<30s}  {grand_total:>5d} imágenes")
    print("═" * 60)
    print(f"\n✓ Imágenes guardadas en: {output_root.resolve()}")


if __name__ == "__main__":
    main()
