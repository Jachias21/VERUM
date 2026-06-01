"""
Pipeline ETL - mantiene actualizada la base de conocimiento de Qdrant con artículos
recientes de fact-checking.

Extrae artículos de fuentes RSS y de la API Google FC, los transforma en vectores
densas (multilingual-e5-large) + sparse (bm25) y los inserta en una colección
híbrida de Qdrant.
"""
from __future__ import annotations

import logging
import os
import socket
import ssl
import uuid

import feedparser
import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

load_dotenv()

logger = logging.getLogger("verum.etl")

from .seed_classics import CLASSIC_HOAXES  # noqa: E402

# Fuentes RSS de fact-checking (ampliar libremente)
RSS_FEEDS = [
    # España - fact-checkers
    "https://maldita.es/feed/",
    "https://newtral.es/feed/",
    "https://verificat.cat/feed/",
    "https://factual.afp.com/list/es/rss.xml",  # possibly down - keep for retry
    "https://www.newtral.es/area/fast-forward/feed/",
    "https://hechosdehoy.com/feed/",
    # América Latina - fact-checkers
    "https://chequeado.com/feed/",          # possibly down - keep for retry
    "https://colombiacheck.com/feed",       # possibly down - keep for retry
    "https://factchequeado.com/feed/",
    "https://lasillavacia.com/feed",
    "https://www.pagina12.com.ar/rss/secciones/el-planeta/notas",  # possibly down - keep for retry
    # Inglés - fact-checkers de alta calidad
    "https://www.snopes.com/feed/",
    "https://www.factcheck.org/feed/",
    "https://fullfact.org/feed/",
    "https://apnews.com/hub/fact-checking/feed",  # possibly down - keep for retry
    "https://www.politifact.com/rss/all.rss/",    # possibly down - keep for retry
    "https://factuel.afp.com/list/en/rss.xml",    # possibly down - keep for retry
    # Organizaciones anti-desinformación
    "https://www.stopfake.org/en/feed/",
    "https://euvsdisinfo.eu/feed/",
    # Medios de referencia con secciones de fact-checking
    "https://elpais.com/rss/elpais/portada.xml",
    "https://www.bbc.com/mundo/rss.xml",
    "https://www.20minutos.es/rss/",
    "https://www.efeverde.com/feed/",
    # Noticias generales de calidad (cobertura amplia)
    "https://www.lavanguardia.com/mvc/feed/rss/vida/salud",  # possibly down - keep for retry
    "https://www.elmundo.es/rss/espana.xml",
    "https://www.rtve.es/api/noticias.rss",         # possibly down - keep for retry
    "https://www.infobae.com/feeds/rss/",           # possibly down - keep for retry
    "https://www.clarin.com/rss/lo-ultimo/",
    # Fuentes institucionales oficiales (España)
    "https://www.lamoncloa.gob.es/serviciosdeprensa/notasprensa/Paginas/index.aspx?format=rss",  # possibly down - keep for retry
    "https://www.boe.es/rss/canal.php?c=ultimas_disposiciones",          # possibly down - keep for retry
    "https://www.sanidad.gob.es/rss/notasPrensa.do",                     # possibly down - keep for retry
    "https://www.educacionyfp.gob.es/prensa/actualidad.rss",             # possibly down - keep for retry
    # Fuentes institucionales oficiales (internacional)
    "https://www.un.org/feed/subscribe/en/rss/category/un-news/feed",  # possibly down - keep for retry
    "https://www.who.int/rss-feeds/news-english.xml",
    "https://www.esa.int/rssfeed/Our_Activities",
    "https://ec.europa.eu/commission/presscorner/api/rss?language=es",
    # Agencias de noticias generales (cobertura factual amplia)
    "https://www.efe.com/efe/espana/1/rss",         # possibly down - keep for retry
    "https://feeds.bbci.co.uk/news/world/rss.xml",  # possibly down - keep for retry
    "https://apnews.com/index.rss",                 # possibly down - keep for retry
    "https://www.dw.com/es/top-stories/s-30684/rss",  # possibly down - keep for retry
    # Ciencia y salud (cobertura con revisión por pares)
    "https://www.nature.com/nature.rss",
    "https://www.nih.gov/news-events/news-releases/feed",  # possibly down - keep for retry
    "https://www.cdc.gov/media/rss/index.htm",             # possibly down - keep for retry
    # Economía (institucional y general de calidad)
    "https://www.bde.es/rss/notas_es.xml",         # possibly down - keep for retry
    "https://cincodias.elpais.com/rss/cincodias/portada.xml",
    "https://www.expansion.com/rss/portada.xml",
]

# Dominios de editores con información factual de fuentes institucionales/oficiales.
# La coincidencia se realiza contra el netloc de la URL del feed (o del artículo para GNews).
# Artículos de estos dominios sin palabras clave FAKE explícitas son REAL por defecto.
_INSTITUTIONAL_DOMAINS = {
    # España - institucional
    "lamoncloa.gob.es",
    "boe.es",
    "ine.es",
    "sanidad.gob.es",
    "educacionyfp.gob.es",
    "exteriores.gob.es",
    "bde.es",
    # Internacional - institucional
    "un.org",
    "who.int",
    "esa.int",
    "ecb.europa.eu",
    "ec.europa.eu",
    "nih.gov",
    "cdc.gov",
    # Agencias de noticias (cobertura factual amplia)
    "efe.com",
    "efeverde.com",      # EFE Verde - environmental section of Spain's national agency
    "reuters.com",
    "bbc.com",
    "bbc.co.uk",
    "bbci.co.uk",
    "apnews.com",
    "dw.com",
    # Editores científicos (cobertura con revisión por pares)
    "nature.com",
    "science.org",
    "sciencemag.org",
    # Quality economy outlets
    "cincodias.elpais.com",
    "expansion.com",
    # España - diarios de referencia de calidad
    "elpais.com",        # El País - Spain's newspaper of record
    "elmundo.es",        # El Mundo - major Spanish daily
    "abc.es",            # ABC - major Spanish daily
    "lavanguardia.com",  # La Vanguardia - major Catalan/Spanish daily
    "20minutos.es",      # 20 Minutos - widely distributed free daily
    "rtve.es",           # RTVE - Spain's public broadcaster
    # América Latina - diarios de referencia de calidad
    "clarin.com",        # Clarín - Argentina's most-read newspaper
    "infobae.com",       # Infobae - major Latin-American digital outlet
    "pagina12.com.ar",   # Página 12 - Argentine daily of record
    "lanacion.com.ar",   # La Nación - Argentina's newspaper of record
}

# Dominios de editores de fact-checking. Coincidencia contra netloc del feed.
# Artículos de estos sin palabras clave FAKE/REAL explícitas son UNVERIFIED por defecto.
_FACTCHECKER_DOMAINS = {
    "maldita.es",
    "newtral.es",
    "verificat.cat",
    "factual.afp.com",
    "factuel.afp.com",
    "hechosdehoy.com",
    "chequeado.com",
    "colombiacheck.com",
    "factchequeado.com",
    "lasillavacia.com",
    "snopes.com",
    "factcheck.org",
    "fullfact.org",
    "politifact.com",
    "stopfake.org",
    "euvsdisinfo.eu",
    "aosfatos.org",
    "ecuadorchequea.com",
}

# Dense vector dimension for BAAI/bge-m3
DENSE_DIM = 1024


_FAKE_KEYWORDS = [
    "falso", "false", "fake", "bulo", "hoax", "incorrecto", "desinformación",
    "engaño", "manipulado", "mentira", "no es cierto", "sin evidencia",
    "misleading", "misinformation", "debunked", "desmentido",
]
_REAL_KEYWORDS = [
    "verdadero", "true", "correcto", "verified", "confirmado", "cierto", "demostrado",
]


from urllib.parse import urlparse


def _domain_class(source_url: str) -> str:
    """Clasifica una URL por dominio del editor. Devuelve 'institutional', 'factchecker' o 'unknown'."""
    if not source_url:
        return "unknown"
    try:
        netloc = urlparse(source_url).netloc.lower()
    except Exception:
        return "unknown"
    if not netloc:
        return "unknown"
    # Eliminar 'www.' inicial para la coincidencia
    if netloc.startswith("www."):
        netloc = netloc[4:]
    # Coincidencia exacta o cualquier subdominio conocido (p.ej. 'feeds.reuters.com' coincide con 'reuters.com')
    for inst in _INSTITUTIONAL_DOMAINS:
        if netloc == inst or netloc.endswith("." + inst):
            return "institutional"
    for fc in _FACTCHECKER_DOMAINS:
        if netloc == fc or netloc.endswith("." + fc):
            return "factchecker"
    return "unknown"


def _infer_verdict_from_entry(
    entry: dict,
    publisher: str = "",
    source_url: str = "",
) -> str:
    """Veredicto best-effort a partir de título, etiquetas, texto y dominio del editor.

    Flujo de decisión:
      1. Las señales léxicas dominan: palabras clave FAKE/REAL explícitas siempre ganan.
      2. Si aparecen ambas → UNVERIFIED (ambiguo).
      3. Sin señales explícitas, se usa la clase de dominio del editor:
         - institutional → REAL.
         - factchecker → UNVERIFIED (contenido puede ser contexto sin veredicto).
         - unknown → UNVERIFIED (valor seguro por defecto).
    """
    tags = " ".join(t.get("term", "") for t in entry.get("tags", [])).lower()
    title = entry.get("title", "").lower()
    raw_summary = entry.get("summary", "")
    clean_summary = BeautifulSoup(raw_summary, "html.parser").get_text(separator=" ", strip=True).lower()
    combined = " ".join([tags, title, clean_summary])

    has_fake = any(w in combined for w in _FAKE_KEYWORDS)
    has_real = any(w in combined for w in _REAL_KEYWORDS)

    domain_class = _domain_class(source_url)

    if has_fake and has_real:
        # Fact-checkers que mencionan ambas señales están desacreditando algo falso
        if domain_class == "factchecker":
            return "FAKE"
        return "UNVERIFIED"
    if has_fake:
        return "FAKE"
    if has_real:
        return "REAL"

    if domain_class == "institutional":
        return "REAL"
    if domain_class == "factchecker":
        return "UNVERIFIED"
    return "UNVERIFIED"


def extract_from_gnews() -> list[dict]:
    """Obtiene artículos de la API GNews para múltiples consultas (español + inglés).
    Devuelve lista vacía si GNEWS_API_KEY no está configurada o la petición falla.
    Deduplica por URL entre consultas.
    """
    api_key = os.getenv("GNEWS_API_KEY", "")
    if not api_key:
        return []

    queries = ["fake news bulo", "desinformación España", "fact check", "hoax viral"]
    seen: dict[str, dict] = {}  # url → article dict (dedup)

    for query in queries:
        url = (
            f"https://gnews.io/api/v4/search"
            f"?q={query.replace(' ', '+')}&lang=es&token={api_key}&max=20"
        )
        try:
            resp = httpx.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"[etl] Error GNews en la consulta {query!r}, omitiendo: {exc}")
            continue

        for item in data.get("articles", []):
            link = item.get("url", "")
            if not link or link in seen:
                continue
            publisher = item.get("source", {}).get("name", "GNews")
            raw_summary = item.get("description", "") or ""
            clean_summary = BeautifulSoup(raw_summary, "html.parser").get_text(separator=" ", strip=True)
            synthetic_entry = {
                "title": item.get("title", ""),
                "summary": item.get("content", raw_summary),
                "tags": [],
            }
            seen[link] = {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, link)),
                "title": item.get("title", ""),
                "summary": clean_summary,
                "url": link,
                "publisher": publisher,
                "published": item.get("publishedAt", ""),
                "verdict": _infer_verdict_from_entry(synthetic_entry, publisher=publisher, source_url=link),
            }

    return list(seen.values())


def _validate_article(article: dict) -> bool:
    """Devuelve True solo si el artículo tiene título, resumen y URL útiles."""
    title = article.get("title", "")
    summary = article.get("summary", "")
    url = article.get("url", "")
    return len(title) >= 10 and len(summary) >= 20 and url.startswith("http")


def extract() -> list[dict]:
    """Obtiene los últimos artículos de los feeds RSS y la API GNews."""
    articles: list[dict] = []
    feeds_ok = 0
    feeds_empty = 0
    feeds_failed = 0
    raw_count = 0

    for url in RSS_FEEDS:
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(15)
        try:
            feed = feedparser.parse(url)
            if not feed.entries:
                feeds_empty += 1
                continue
            feeds_ok += 1
            publisher = feed.feed.get("title", "")
            for entry in feed.entries:
                raw_summary = entry.get("summary", "")
                clean_summary = BeautifulSoup(raw_summary, "html.parser").get_text(separator=" ", strip=True)
                article = {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, entry.get("link", ""))),
                    "title": entry.get("title", ""),
                    "summary": clean_summary,
                    "url": entry.get("link", ""),
                    "publisher": publisher,
                    "published": entry.get("published", ""),
                    "verdict": _infer_verdict_from_entry(entry, publisher=publisher, source_url=url),
                }
                raw_count += 1
                if _validate_article(article):
                    articles.append(article)
        except (socket.timeout, TimeoutError):
            print(f"[etl] TIMEOUT: {url}")
            feeds_failed += 1
        except ssl.SSLError:
            print(f"[etl] ERROR_SSL: {url}")
            feeds_failed += 1
        except Exception as exc:
            print(f"[etl] FALLIDO ({type(exc).__name__}): {url} - {exc}")
            feeds_failed += 1
        finally:
            socket.setdefaulttimeout(old_timeout)

    discarded = raw_count - len(articles)
    if discarded:
        print(f"[etl] Descartados {discarded} artículos con título/resumen/url vacíos")
    print(f"[etl] Resumen de feeds: {feeds_ok} OK / {feeds_empty} vacíos / {feeds_failed} fallidos")

    gnews_articles = extract_from_gnews()
    if gnews_articles:
        print(f"[etl] GNews aportó {len(gnews_articles)} artículos (entre todas las consultas).")
    articles.extend(gnews_articles)
    return articles


BATCH_SIZE = 10  # max articles per embedding + upsert cycle to avoid OOM


def _ensure_collection_schema(client: QdrantClient, collection: str, force_recreate: bool = False) -> None:
    """Asegura que la colección Qdrant exista con el esquema híbrido correcto.
    Por defecto usa modo upsert incremental: si la colección ya existe se deja intacta.
    Pasar force_recreate=True solo para migraciones de esquema.
    """
    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        if force_recreate:
            print(f"[etl] force_recreate=True - eliminando colección existente '{collection}'...")
            client.delete_collection(collection)
        else:
            print(f"[etl] La colección ya existe, usando modo upsert")
            return
    print(f"[etl] Creando colección híbrida '{collection}' (densa + sparse)...")
    client.create_collection(
        collection_name=collection,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            ),
        },
    )


def _embed_batch(
    articles: list[dict],
    dense_model,
    sparse_model,
) -> list[PointStruct]:
    """Genera embeddings de un lote y devuelve PointStructs. No carga modelos."""
    import gc

    texts = [f"{art['title']} {art['summary']}" for art in articles]

    dense_vectors = list(dense_model.embed(texts))
    sparse_embeddings = list(sparse_model.embed(texts))

    points = []
    for art, dense_vec, sparse_emb in zip(articles, dense_vectors, sparse_embeddings):
        points.append(
            PointStruct(
                id=art["id"],
                vector={
                    "dense": dense_vec.tolist(),
                    "sparse": SparseVector(
                        indices=sparse_emb.indices.tolist(),
                        values=sparse_emb.values.tolist(),
                    ),
                },
                payload={k: v for k, v in art.items() if k != "id"},
            )
        )

    # Liberar tensores intermedios
    del texts, dense_vectors, sparse_embeddings
    gc.collect()
    return points


# Se mantienen transform() y load() para llamadas directas desde scripts legacy.
def transform(articles: list[dict]) -> list[PointStruct]:
    """Genera embeddings dense + sparse para cada artículo. Devuelve lista de PointStruct."""
    from fastembed import TextEmbedding, SparseTextEmbedding
    import gc

    print("[etl] Cargando modelo denso (multilingual-e5-large, ONNX)...")
    dense_model = TextEmbedding(model_name="intfloat/multilingual-e5-large", threads=2)
    print("[etl] Cargando modelo sparse (Qdrant/bm25)...")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25", threads=2)

    points = _embed_batch(articles, dense_model, sparse_model)

    del dense_model, sparse_model
    gc.collect()
    return points


def load(points: list[PointStruct]) -> None:
    """Asegura el esquema de la colección e inserta todos los puntos (modo incremental por defecto)."""
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )
    collection = os.getenv("QDRANT_COLLECTION", "fact_checks")
    _ensure_collection_schema(client, collection)
    client.upsert(collection_name=collection, points=points)
    print(f"[etl] Insertados {len(points)} puntos en '{collection}' (modo incremental).")


def run() -> None:
    import gc
    from fastembed import TextEmbedding, SparseTextEmbedding

    print("[etl] Iniciando pipeline ETL híbrido...")
    articles = extract()
    print(f"[etl] Extraídos {len(articles)} artículos (válidos).")

    # Setup Qdrant (collection created only if absent; upsert mode otherwise)
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )
    collection = os.getenv("QDRANT_COLLECTION", "fact_checks")
    _ensure_collection_schema(client, collection)

    # Seed classic hoaxes if requested or collection is nearly empty
    _seed_flag = os.getenv("SEED_CLASSICS", "true").strip().lower()
    _should_seed = _seed_flag in ("1", "true", "yes")
    if not _should_seed:
        try:
            _point_count = client.count(collection_name=collection).count
            _should_seed = _point_count < 5
        except Exception:
            _should_seed = False
    if _should_seed:
        logger.info("[etl] Seeding %d classic hoaxes into Qdrant", len(CLASSIC_HOAXES))
        print(f"[etl] Seeding {len(CLASSIC_HOAXES)} classic hoaxes into Qdrant...")
        seed_articles = [
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, h["url"])),
                "title": h["title"],
                "summary": h["content_summary"],
                "url": h["url"],
                "publisher": h["source_publisher"],
                "published": h["publish_date"],
                "verdict": h["verdict"],
            }
            for h in CLASSIC_HOAXES
        ]
        articles = seed_articles + articles

    # Load models once - kept alive across batches
    print("[etl] Cargando modelo denso (multilingual-e5-large, ONNX)...")
    dense_model = TextEmbedding(model_name="intfloat/multilingual-e5-large")

    print("[etl] Cargando modelo sparse (Qdrant/bm25)...")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    total = len(articles)
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    total_upserted = 0

    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        batch = articles[start: start + BATCH_SIZE]
        print(f"[etl] Procesando y guardando lote {batch_idx + 1}/{num_batches} "
              f"({len(batch)} artículos)...")

        points = _embed_batch(batch, dense_model, sparse_model)
        client.upsert(collection_name=collection, points=points)
        total_upserted += len(points)

        del points
        gc.collect()

    # Release models
    del dense_model, sparse_model
    gc.collect()

    print(f"[etl] Insertados {total_upserted} puntos en '{collection}' (modo incremental).")

    from collections import Counter
    counts = Counter(art["publisher"] for art in articles)
    print("[etl] Artículos por editor (top 10):")
    for publisher, count in counts.most_common(10):
        print(f"  {count:>5}  {publisher}")

    print("[etl] Completado.")


if __name__ == "__main__":
    run()
