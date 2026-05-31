#!/usr/bin/env python3
"""
download_raise.py — Descarga imágenes TIFF de RAISE-2k desde el CSV oficial.

El CSV de RAISE-2k contiene una columna con la URL de cada TIFF.
Este script descarga las imágenes a un directorio local de forma robusta:
  · Reintenta hasta 3 veces por imagen en caso de error de red
  · Omite imágenes ya descargadas (idempotente)
  · Muestra barra de progreso y resumen final

Uso:
  python scripts/download_raise.py \\
      --csv-path  data/raw/raise/RAISE_all.csv \\
      --output-dir data/raw/raise/images \\
      --max-images 2000

Dependencias: requests, pandas
"""

import argparse
import sys
import time
from pathlib import Path

# ── Importaciones opcionales con mensajes claros ───────────────────────────────
try:
    import pandas as pd
except ImportError:
    sys.exit("[ERROR] Instala pandas:    pip install pandas")

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    sys.exit("[ERROR] Instala requests:  pip install requests")


# ── Constantes ─────────────────────────────────────────────────────────────────
# Candidatos de nombre de columna URL (probados en orden)
URL_COLUMN_CANDIDATES = ["TIFF", "tiff", "url", "URL", "image_url", "link", "Path"]

TIMEOUT_SECONDS   = 30       # timeout por petición
MAX_RETRIES       = 3        # reintentos ante error de red
BACKOFF_FACTOR    = 1.5      # espera entre reintentos (exponencial)
CHUNK_SIZE        = 1 << 16  # 64 KB por chunk


# ── Utilidades ─────────────────────────────────────────────────────────────────

def _build_session() -> requests.Session:
    """Crea una sesión requests con retry automático."""
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers["User-Agent"] = "RAISE-downloader/1.0 (research use)"
    return session


def _find_url_column(columns: list[str]) -> str:
    """Devuelve el nombre de la columna URL encontrada en el CSV."""
    for c in URL_COLUMN_CANDIDATES:
        if c in columns:
            return c
    raise ValueError(
        f"No se encontró columna de URL. "
        f"Columnas disponibles: {columns}. "
        f"Candidatos esperados: {URL_COLUMN_CANDIDATES}"
    )


def _progress_bar(current: int, total: int, width: int = 40) -> str:
    """Barra de progreso ASCII."""
    pct  = current / total if total else 0
    done = int(width * pct)
    bar  = "█" * done + "░" * (width - done)
    return f"[{bar}] {current}/{total} ({pct*100:.1f}%)"


def _safe_filename(url: str, idx: int) -> str:
    """Extrae el nombre de fichero de la URL; usa índice como fallback."""
    name = url.rstrip("/").split("/")[-1].split("?")[0]
    if not name or "." not in name:
        name = f"raise_{idx:05d}.tiff"
    return name


# ── Descarga principal ─────────────────────────────────────────────────────────

def download(
    csv_path: Path,
    output_dir: Path,
    max_images: int,
) -> dict[str, int]:
    """
    Lee el CSV y descarga hasta max_images TIFFs.

    Returns:
        dict con contadores {"downloaded": n, "skipped": n, "errors": n}
    """
    print(f"\n{'='*62}")
    print("  RAISE-2k — Descarga de imágenes TIFF")
    print(f"{'='*62}")
    print(f"  CSV     : {csv_path}")
    print(f"  Salida  : {output_dir}")
    print(f"  Límite  : {max_images:,} imágenes")
    print(f"{'='*62}\n")

    # Leer CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        sys.exit(f"[ERROR] No se pudo leer el CSV: {exc}")

    print(f"  Filas en el CSV : {len(df):,}")

    # Detectar columna URL
    try:
        url_col = _find_url_column(df.columns.tolist())
    except ValueError as exc:
        sys.exit(f"[ERROR] {exc}")

    print(f"  Columna URL     : '{url_col}'\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    session   = _build_session()
    counters  = {"downloaded": 0, "skipped": 0, "errors": 0}
    t_start   = time.time()
    rows      = df[url_col].dropna().tolist()
    total     = min(len(rows), max_images)

    for idx, url in enumerate(rows):
        if counters["downloaded"] + counters["skipped"] >= max_images:
            break

        url = str(url).strip()
        if not url.startswith("http"):
            counters["errors"] += 1
            print(f"  ⚠ [{idx+1}] URL inválida: {url!r}")
            continue

        filename = _safe_filename(url, idx)
        dest     = output_dir / filename

        # ── Omitir si ya existe ───────────────────────────────────────────────
        if dest.exists() and dest.stat().st_size > 0:
            counters["skipped"] += 1
            _print_progress(counters, total, t_start)
            continue

        # ── Descargar ─────────────────────────────────────────────────────────
        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with session.get(url, timeout=TIMEOUT_SECONDS, stream=True) as resp:
                    resp.raise_for_status()
                    with open(dest, "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                            fh.write(chunk)
                success = True
                break
            except requests.exceptions.RequestException as exc:
                if attempt < MAX_RETRIES:
                    wait = BACKOFF_FACTOR ** attempt
                    time.sleep(wait)
                else:
                    counters["errors"] += 1
                    # Eliminar fichero parcial
                    if dest.exists():
                        dest.unlink(missing_ok=True)
                    err_short = str(exc)[:80]
                    print(f"\n  ⚠ [{idx+1}/{total}] Error tras {MAX_RETRIES} intentos: {err_short}")

        if success:
            counters["downloaded"] += 1

        _print_progress(counters, total, t_start)

    return counters


def _print_progress(counters: dict, total: int, t_start: float) -> None:
    done  = counters["downloaded"] + counters["skipped"]
    bar   = _progress_bar(done, total)
    elapsed = time.time() - t_start
    rate  = done / elapsed if elapsed > 0 else 0
    print(f"  {bar}  {rate:.1f} img/s", end="\r", flush=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga imágenes TIFF de RAISE-2k desde su CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        required=True,
        help="Ruta al CSV de RAISE-2k (p.ej. RAISE_all.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directorio donde se guardarán los ficheros TIFF.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=2000,
        metavar="N",
        help="Número máximo de imágenes a descargar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.csv_path.exists():
        sys.exit(f"[ERROR] El CSV no existe: {args.csv_path}")
    if args.max_images < 1:
        sys.exit("[ERROR] --max-images debe ser >= 1")

    t0       = time.time()
    counters = download(args.csv_path, args.output_dir, args.max_images)
    elapsed  = time.time() - t0
    total    = counters["downloaded"] + counters["skipped"]

    # ── Resumen final ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*62}")
    print("  RESUMEN FINAL")
    print(f"{'='*62}")
    print(f"  ✓ Descargadas nuevas   : {counters['downloaded']:>7,}")
    print(f"  ○ Omitidas (ya existían): {counters['skipped']:>7,}")
    print(f"  ⚠ Errores              : {counters['errors']:>7,}")
    print(f"  ─ Total procesadas     : {total:>7,}")
    print(f"  ⏱ Tiempo total         : {elapsed:>6.1f}s")
    if elapsed > 0 and total > 0:
        print(f"  ⚡ Velocidad media      : {total/elapsed:>6.1f} img/s")
    print(f"{'='*62}")
    print(f"  Salida → {args.output_dir.resolve()}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
