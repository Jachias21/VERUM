"""
scrape_real.py — descarga imágenes reales de Unsplash y Wikimedia Commons
para la clase REAL del dataset propio de VERUM.

Fuentes:
    1. Unsplash API — endpoint /search/photos, per_page=30.
                      Requiere --unsplash-key o UNSPLASH_ACCESS_KEY.
                      Rate limit: 50 req/hora → delay de 72s entre requests.
    2. Wikimedia Commons API — gratuita, sin key, fuente complementaria o
                               fallback si no hay Unsplash key.

Uso:
    python scripts/build_dataset/scrape_real.py
    python scripts/build_dataset/scrape_real.py \\
        --unsplash-key TU_KEY \\
        --output-dir data/custom/real/ \\
        --images-per-query 30

Dependencias:
    pip install requests Pillow
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Queries predefinidas agrupadas por categoría
# ---------------------------------------------------------------------------
QUERIES: dict[str, list[str]] = {
    "personas_famosas": [
        "famous politicians portrait",
        "celebrities red carpet",
        "world leaders press photo",
        "famous musicians live concert",
        "athletes sports competition",
    ],
    "eventos_sociales": [
        "protest demonstration crowd",
        "natural disaster flood earthquake",
        "political rally crowd",
        "street festival celebration",
        "humanitarian crisis refugee",
    ],
    "paisajes_lugares": [
        "mountain landscape nature",
        "urban cityscape street",
        "ocean beach sunset",
        "forest river waterfall",
        "desert dunes aerial",
    ],
    "animales": [
        "wildlife animals nature",
        "dogs cats pets",
        "birds flying nature",
        "ocean fish marine life",
        "farm animals countryside",
    ],
    "comida": [
        "restaurant food plated meal",
        "fresh fruit vegetables market",
        "street food vendor",
        "bakery bread pastry",
        "traditional cuisine dish",
    ],
    "objetos_cotidianos": [
        "everyday objects desk workspace",
        "tools hardware workshop",
        "electronics gadgets",
        "clothing accessories fashion",
        "books stationery office",
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
# Unsplash rate-limit
# Tier gratuito: 50 requests/hora → mínimo 72s entre cada request a la API.
# ---------------------------------------------------------------------------
_UNSPLASH_API = "https://api.unsplash.com"
_UNSPLASH_RATE_DELAY = 72.0   # segundos entre requests (3600 / 50)

# ---------------------------------------------------------------------------
# Helpers genéricos
# ---------------------------------------------------------------------------

def _safe_dirname(query: str) -> str:
    """Convierte un query arbitrario en un nombre de directorio válido."""
    name = query.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name[:60].strip("_")


def _get_requests():
    """Importa requests o aborta con mensaje claro."""
    try:
        import requests  # noqa: PLC0415
        return requests
    except ImportError:
        log.error("'requests' no instalado. Instálalo con:  pip install requests")
        sys.exit(1)


def _download_file(url: str, dest: Path, headers: Optional[dict] = None) -> bool:
    """Descarga una URL en ``dest``. Devuelve True si tuvo éxito."""
    requests = _get_requests()
    try:
        resp = requests.get(url, timeout=20, stream=True, headers=headers or {})
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except Exception as exc:  # noqa: BLE001
        log.debug("Error descargando %s: %s", url, exc)
        return False


def _image_is_valid(path: Path, min_px: int) -> bool:
    """Devuelve True si la imagen puede abrirse y supera min_px×min_px."""
    try:
        from PIL import Image  # noqa: PLC0415
        with Image.open(path) as im:
            w, h = im.size
        return w >= min_px and h >= min_px
    except Exception:  # noqa: BLE001
        return False


def _has_camera_exif(path: Path) -> bool:
    """Devuelve True si la imagen contiene metadatos EXIF de cámara."""
    try:
        from PIL import Image          # noqa: PLC0415
        from PIL.ExifTags import TAGS  # noqa: PLC0415
        with Image.open(path) as im:
            raw_exif = im._getexif()   # type: ignore[attr-defined]
        if not raw_exif:
            return False
        exif = {TAGS.get(k, k): v for k, v in raw_exif.items()}
        camera_tags = {"Make", "Model", "LensMake", "LensModel"}
        return bool(camera_tags & exif.keys())
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Unsplash API
# ---------------------------------------------------------------------------

def _unsplash_search(api_key: str, query: str, page: int = 1) -> list[dict]:
    """
    Llama a GET /search/photos con per_page=30 y orientation=squarish.
    Devuelve la lista de objetos foto de Unsplash.
    """
    requests = _get_requests()
    url = f"{_UNSPLASH_API}/search/photos"
    params = {
        "query":       query,
        "per_page":    30,
        "orientation": "squarish",
        "page":        page,
    }
    headers = {"Authorization": f"Client-ID {api_key}"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception as exc:  # noqa: BLE001
        log.warning("Unsplash search error (query='%s', page=%d): %s", query, page, exc)
        return []


def _crawl_unsplash(
    api_key: str,
    query: str,
    dest: Path,
    max_num: int,
    min_px: int,
) -> tuple[int, int]:
    """
    Descarga hasta ``max_num`` imágenes de Unsplash para ``query``.

    Respeta el rate limit de 50 req/hora aplicando un delay de 72s
    entre cada llamada a la API de búsqueda.

    Returns
    -------
    (downloaded, with_camera_exif)
    """
    dest.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    with_exif  = 0
    page       = 1

    while downloaded < max_num:
        log.debug("    [Unsplash] query='%s' page=%d", query, page)
        photos = _unsplash_search(api_key, query, page=page)
        if not photos:
            log.info("    [Unsplash] Sin más resultados para '%s'.", query)
            break

        for photo in photos:
            if downloaded >= max_num:
                break

            photo_id = photo.get("id", "")
            url      = (photo.get("urls") or {}).get("regular")
            if not url or not photo_id:
                continue

            filename = dest / f"unsplash_{photo_id}.jpg"
            if filename.exists():
                downloaded += 1   # ya descargada en una ejecución anterior
                if _has_camera_exif(filename):
                    with_exif += 1
                continue

            ok = _download_file(url, filename)
            if not ok:
                continue

            if not _image_is_valid(filename, min_px):
                filename.unlink(missing_ok=True)
                continue

            if _has_camera_exif(filename):
                with_exif += 1
            downloaded += 1

        page += 1

        # ── Rate-limit: máximo 50 requests/hora ──────────────────────────────
        if downloaded < max_num and photos:
            log.info(
                "    [Unsplash] Esperando %.0fs para respetar rate limit "
                "(50 req/hora)…",
                _UNSPLASH_RATE_DELAY,
            )
            time.sleep(_UNSPLASH_RATE_DELAY)

    return downloaded, with_exif


# ---------------------------------------------------------------------------
# Wikimedia Commons API
# ---------------------------------------------------------------------------
_WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
_WIKIMEDIA_HEADERS = {"User-Agent": "VERUM-Dataset-Builder/1.0 (research project)"}
_VALID_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def _wikimedia_search(query: str, limit: int = 50, offset: int = 0) -> list[str]:
    """
    Busca en Wikimedia Commons y devuelve lista de títulos (p.ej. 'File:Foo.jpg').
    """
    requests = _get_requests()
    params = {
        "action":      "query",
        "list":        "search",
        "srsearch":    f"{query} filetype:bitmap",
        "srnamespace": 6,          # NS_FILE
        "srlimit":     min(limit, 50),
        "sroffset":    offset,
        "format":      "json",
    }
    try:
        resp = requests.get(
            _WIKIMEDIA_API, params=params, timeout=10,
            headers=_WIKIMEDIA_HEADERS,
        )
        resp.raise_for_status()
        return [r["title"] for r in resp.json().get("query", {}).get("search", [])]
    except Exception as exc:  # noqa: BLE001
        log.warning("Wikimedia search error: %s", exc)
        return []


def _wikimedia_image_url(title: str) -> Optional[str]:
    """Resuelve el título de un File: de Wikimedia a su URL de descarga directa."""
    requests = _get_requests()
    params = {
        "action": "query",
        "titles": title,
        "prop":   "imageinfo",
        "iiprop": "url",
        "format": "json",
    }
    try:
        resp = requests.get(
            _WIKIMEDIA_API, params=params, timeout=10,
            headers=_WIKIMEDIA_HEADERS,
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            info = page.get("imageinfo", [])
            if info:
                return info[0].get("url")
    except Exception as exc:  # noqa: BLE001
        log.debug("Wikimedia imageinfo error (%s): %s", title, exc)
    return None


def _crawl_wikimedia(
    query: str,
    dest: Path,
    max_num: int,
    min_px: int,
    delay: float,
) -> tuple[int, int]:
    """
    Descarga hasta ``max_num`` imágenes de Wikimedia Commons para ``query``.

    Returns
    -------
    (downloaded, with_camera_exif)
    """
    dest.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    with_exif  = 0
    offset     = 0

    while downloaded < max_num:
        titles = _wikimedia_search(query, limit=50, offset=offset)
        if not titles:
            break

        for title in titles:
            if downloaded >= max_num:
                break

            url = _wikimedia_image_url(title)
            if not url:
                continue

            ext = Path(url.split("?")[0]).suffix.lower()
            if ext not in _VALID_IMG_EXT:
                continue

            safe_name = re.sub(r"[^a-z0-9_.-]", "_", title.lower())[:80]
            filename  = dest / f"wiki_{safe_name}"
            if not filename.suffix:
                filename = filename.with_suffix(ext)
            if filename.exists():
                continue

            ok = _download_file(url, filename)
            if not ok:
                continue

            if not _image_is_valid(filename, min_px):
                filename.unlink(missing_ok=True)
                continue

            if _has_camera_exif(filename):
                with_exif += 1
            downloaded += 1
            time.sleep(0.3)   # cortesía con la API de Wikimedia

        offset += 50
        time.sleep(delay)

    return downloaded, with_exif


# ---------------------------------------------------------------------------
# Dispatcher principal
# ---------------------------------------------------------------------------

def _crawl_query(
    query: str,
    dest: Path,
    max_num: int,
    min_px: int,
    delay: float,
    unsplash_key: Optional[str],
) -> tuple[int, int]:
    """
    Descarga imágenes para un query.

    Estrategia:
      - Si hay ``unsplash_key``: usa Unsplash como fuente primaria y
        complementa con Wikimedia si faltan imágenes.
      - Sin key: usa solo Wikimedia Commons.

    Returns
    -------
    (downloaded, with_camera_exif)
    """
    if unsplash_key:
        log.debug("    [Unsplash] '%s'", query)
        n_dl, n_exif = _crawl_unsplash(unsplash_key, query, dest, max_num, min_px)

        # Complementar con Wikimedia si Unsplash no llegó al objetivo
        if n_dl < max_num:
            remaining = max_num - n_dl
            log.info(
                "    [Wikimedia] Complementando %d imágenes restantes para '%s'…",
                remaining, query,
            )
            w_dl, w_exif = _crawl_wikimedia(query, dest, remaining, min_px, delay)
            n_dl   += w_dl
            n_exif += w_exif

        return n_dl, n_exif
    else:
        log.debug("    [Wikimedia] '%s'", query)
        return _crawl_wikimedia(query, dest, max_num, min_px, delay)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Descarga imágenes reales de Unsplash / Wikimedia Commons "
            "para el dataset VERUM."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--unsplash-key",
        default=os.environ.get("UNSPLASH_ACCESS_KEY"),
        metavar="KEY",
        help=(
            "Unsplash Access Key. Si no se proporciona se usa la variable de "
            "entorno UNSPLASH_ACCESS_KEY. Sin key, se usa solo Wikimedia Commons."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/custom/real"),
        help="Directorio raíz donde se guardan las imágenes descargadas.",
    )
    parser.add_argument(
        "--images-per-query",
        type=int,
        default=30,
        help=(
            "Número máximo de imágenes a descargar por cada query. "
            "Con Unsplash (per_page=30) un solo request cubre el default."
        ),
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=256,
        help="Tamaño mínimo (px) en cada dimensión. Imágenes menores se descartan.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help=(
            "Pausa en segundos entre queries de Wikimedia. "
            "El delay de Unsplash (72s) se aplica siempre automáticamente."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    output_root:      Path           = args.output_dir
    images_per_query: int            = args.images_per_query
    min_size:         int            = args.min_size
    delay:            float          = args.delay
    unsplash_key:     Optional[str]  = args.unsplash_key

    output_root.mkdir(parents=True, exist_ok=True)

    if unsplash_key:
        log.info(
            "Fuente primaria: Unsplash API (key=%s…) + Wikimedia como complemento.",
            unsplash_key[:6],
        )
        log.info(
            "⚠  Rate limit Unsplash: 50 req/hora → %.0fs de espera entre páginas.",
            _UNSPLASH_RATE_DELAY,
        )
    else:
        log.info(
            "Fuente: Wikimedia Commons (sin UNSPLASH_ACCESS_KEY). "
            "Para mejores resultados, pasa --unsplash-key o define UNSPLASH_ACCESS_KEY."
        )

    # summary: category → (total_downloaded, total_with_exif)
    summary: dict[str, tuple[int, int]] = {}
    total_categories = len(QUERIES)

    for cat_idx, (category, queries) in enumerate(QUERIES.items(), start=1):
        cat_dir = output_root / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "━━━ Categoría %d/%d: %s (%d queries) ━━━",
            cat_idx, total_categories, category, len(queries),
        )

        cat_downloaded = cat_exif = 0

        for q_idx, query in enumerate(queries, start=1):
            query_dir = cat_dir / _safe_dirname(query)
            log.info(
                "  [%d/%d] '%s'  →  %s",
                q_idx, len(queries), query, query_dir,
            )

            n_dl, n_exif = _crawl_query(
                query=query,
                dest=query_dir,
                max_num=images_per_query,
                min_px=min_size,
                delay=delay,
                unsplash_key=unsplash_key,
            )
            cat_downloaded += n_dl
            cat_exif       += n_exif

            log.info(
                "         ✓ %d descargadas  |  %d con EXIF de cámara",
                n_dl, n_exif,
            )

            if q_idx < len(queries):
                time.sleep(delay)

        summary[category] = (cat_downloaded, cat_exif)

        if cat_idx < total_categories:
            log.info("  Pausa entre categorías (%.1fs)…", delay * 2)
            time.sleep(delay * 2)

    # ── Resumen final ────────────────────────────────────────────────────────
    grand_total = 0
    grand_exif  = 0

    print("\n" + "═" * 68)
    print("  RESUMEN — imágenes reales descargadas")
    print("═" * 68)
    print(f"  {'Categoría':<30s}  {'Descargadas':>12s}  {'Con EXIF':>10s}")
    print("─" * 68)
    for category, (n_dl, n_exif) in summary.items():
        print(f"  {category:<30s}  {n_dl:>12d}  {n_exif:>10d}")
        grand_total += n_dl
        grand_exif  += n_exif
    print("─" * 68)
    exif_pct = (grand_exif / grand_total * 100) if grand_total else 0.0
    print(f"  {'TOTAL':<30s}  {grand_total:>12d}  {grand_exif:>10d}")
    print("═" * 68)
    print(
        f"\n  Calidad EXIF : {grand_exif}/{grand_total} imágenes con metadatos "
        f"de cámara ({exif_pct:.1f}%)"
    )
    print(f"  Tamaño mínimo: {min_size}×{min_size} px")
    if unsplash_key:
        print("  Fuente primaria: Unsplash API + Wikimedia Commons (complemento)")
    else:
        print("  Fuente: Wikimedia Commons")
    print(f"\n✓ Imágenes guardadas en: {output_root.resolve()}")
    print("═" * 68)


if __name__ == "__main__":
    main()
