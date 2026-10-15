"""Servicio de recuperación para la API: FTS, denso o híbrido (RRF), con inquilinos opcionales.

Usa exactamente el mismo código que `evaluate` (mismos índices, misma profundidad de 200 chunks y misma conversión a
artículos distintos), de modo que lo que devuelve el endpoint es lo que se midió. Es síncrono: la API lo ejecuta en
un hilo (`asyncio.to_thread`) para no bloquear el bucle de eventos con la base de datos ni con el modelo.
"""

import threading
from dataclasses import dataclass

import psycopg

from scripts import db, embeddings
from scripts.dense import DenseIndex
from scripts.download_corpus import sha256_bytes
from scripts.evaluate import CHUNK_FILES, DEPTH
from scripts.fts import FtsIndex
from scripts.hybrid import HybridIndex

MODES = ("hybrid", "fts", "dense")
SNIPPET_CHARS = 400
MAX_CACHED_INDEXES = 64  # la cabecera del inquilino la controla el cliente: la caché no puede crecer sin límite

HYDRATE_SQL = "SELECT chunk_id, act, celex, article, title, text FROM chunks WHERE chunk_id = ANY(%s)"
PUBLIC_ONLY = " AND tenant_id IS NULL"  # sin inquilino: solo el corpus público


class ServiceUnavailable(Exception):
    """Base de los fallos que la API traduce a 503."""


class ModelUnavailable(ServiceUnavailable):
    """El modo pedido necesita embeber la consulta y el modelo no está disponible."""


class DatabaseNotReady(ServiceUnavailable):
    """La base de datos no responde, no está cargada o está desfasada respecto a data/chunks."""


@dataclass(frozen=True)
class SearchHit:
    rank: int
    chunk_id: str
    act: str
    celex: str
    article: int
    title: str
    snippet: str


def make_snippet(chunk_text: str, limit: int = SNIPPET_CHARS) -> str:
    """Cuerpo del chunk sin la cabecera «Article N. Título», en una línea y recortado en una frontera de palabra."""
    body = chunk_text.split("\n", 1)[1] if "\n" in chunk_text else chunk_text
    body = " ".join(body.split())
    if len(body) <= limit:
        return body
    return body[:limit].rsplit(" ", 1)[0].rstrip(" ,;:.") + "…"


class RetrievalService:
    def __init__(self, schema: str | None = None, embed_query=None, enable_tenants: bool = False):
        self.schema = schema
        self.tenants_enabled = enable_tenants
        self.model_error: str | None = None
        self._embed_query = embed_query  # None: el modelo se carga en `start()`
        self._model = None
        self._model_lock = threading.Lock()  # la inferencia con un mismo objeto desde varios hilos no se da por segura
        self._cache_lock = threading.Lock()
        self._indexes: dict[tuple, object] = {}

    # ---------- arranque y estado ----------

    def start(self) -> None:
        """Intenta cargar el modelo. Si no puede (sin torch, sin red…), FTS sigue funcionando."""
        if self._embed_query is not None:
            return
        try:
            self._model = embeddings.load_model()
            self._embed_query = self._encode
        except (ImportError, RuntimeError, OSError) as error:
            self.model_error = f"{type(error).__name__}: {error}"

    def _encode(self, query: str):
        with self._model_lock:
            return embeddings.encode(self._model, embeddings.texts_for("queries", [{"query": query}]))[0]

    def status(self) -> dict:
        try:
            with db.connect(self.schema) as conn:
                conn.execute("SELECT 1")
            database = True
        except (psycopg.Error, RuntimeError):
            database = False
        return {"database": database, "model": self._embed_query is not None, "tenants_enabled": self.tenants_enabled}

    # ---------- búsqueda ----------

    def _index(self, mode: str, chunking: str, tenant: str | None):
        key = (mode, chunking, tenant)
        with self._cache_lock:
            if key in self._indexes:
                return self._indexes[key]
        sha = sha256_bytes(CHUNK_FILES[chunking].read_bytes())
        if mode == "fts":
            index = FtsIndex(chunking, sha, self.schema, tenant)
        elif mode == "dense":
            index = DenseIndex(chunking, sha, self._embed_query, self.schema, tenant)
        else:
            index = HybridIndex(chunking, sha, self._embed_query, self.schema, tenant)
        with self._cache_lock:
            if len(self._indexes) >= MAX_CACHED_INDEXES:
                self._indexes.pop(next(iter(self._indexes)))  # el más antiguo
            self._indexes[key] = index
        return index

    def search(self, query: str, top_k: int = 10, mode: str = "hybrid", chunking: str = "window", tenant: str | None = None) -> list[SearchHit]:
        if mode not in MODES:
            raise ValueError(f"modo desconocido: {mode!r}")
        if chunking not in CHUNK_FILES:
            raise ValueError(f"chunking desconocido: {chunking!r}")
        tenant = tenant if self.tenants_enabled else None  # sin inquilinos habilitados, el parámetro se ignora aquí también
        if mode in ("dense", "hybrid") and self._embed_query is None:
            raise ModelUnavailable(self.model_error or "modelo de embeddings no cargado")
        try:
            chunks = self._index(mode, chunking, tenant).search_chunks(query, DEPTH)
            return self._hydrate(chunks, top_k, tenant)
        except (psycopg.Error, RuntimeError) as error:
            raise DatabaseNotReady(str(error)) from error

    def _hydrate(self, chunks: list[tuple[str, tuple[str, int]]], top_k: int, tenant: str | None) -> list[SearchHit]:
        """Un resultado por artículo (el de su mejor chunk, que es el primero en el ranking), con su título y fragmento."""
        best: dict[tuple[str, int], str] = {}
        for chunk_id, article in chunks:
            best.setdefault(article, chunk_id)
            if len(best) == top_k:
                break
        if not best:
            return []
        sql = HYDRATE_SQL + ("" if tenant is not None else PUBLIC_ONLY)
        with db.connect(self.schema) as conn:
            if tenant is not None:
                db.tenant_transaction(conn, tenant)  # RLS decide qué filas se ven
            rows = {r[0]: r for r in conn.execute(sql, (list(best.values()),)).fetchall()}
        hits = []
        for chunk_id in best.values():
            if chunk_id in rows:  # defensivo: la fila pudo desaparecer entre la búsqueda y la lectura
                _, act, celex, article, title, text = rows[chunk_id]
                hits.append(SearchHit(len(hits) + 1, chunk_id, act, celex, article, title, make_snippet(text)))
        return hits
