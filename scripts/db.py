"""Conexión a Postgres, esquema y carga de chunks.

La configuración sale del entorno (.env): POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB y, opcionalmente,
POSTGRES_HOST (por defecto 127.0.0.1) y POSTGRES_PORT (por defecto 5433, el que publica docker-compose).
"""

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv

from scripts.download_corpus import ROOT, sha256_bytes

SCHEMA_FILE = ROOT / "sql" / "schema.sql"
# campo del JSONL -> columna de la tabla ('window' es palabra reservada de SQL)
CHUNK_COLUMNS = {
    "chunk_id": "chunk_id",
    "strategy": "strategy",
    "celex": "celex",
    "act": "act",
    "article": "article",
    "title": "title",
    "chapter_id": "chapter_id",
    "window": "window_index",
    "word_start": "word_start",
    "word_end": "word_end",
    "text": "text",
    "n_words": "n_words",
}


def conninfo() -> dict:
    load_dotenv(ROOT / ".env")
    missing = [v for v in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB") if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"faltan variables en .env: {', '.join(missing)} (ver .env.example)")
    return {
        "host": os.getenv("POSTGRES_HOST") or "127.0.0.1",
        "port": int(os.getenv("POSTGRES_PORT") or 5433),
        "user": os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
        "dbname": os.environ["POSTGRES_DB"],
    }


@contextmanager
def connect(schema: str | None = None):
    """Conexión con `with`: confirma al salir sin error y deshace si hay excepción. `schema` aísla las tablas
    (se usa en los tests para no tocar los datos de desarrollo)."""
    options = f"-c search_path={schema},public" if schema else None
    with psycopg.connect(**conninfo(), options=options) as conn:
        yield conn


def init_db(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA_FILE.read_text(encoding="utf-8"))


def to_vector_literal(vector) -> str:
    """'[0.1,0.2,…]' con el valor exacto de cada float32 (se escribe como float64, que lo representa sin pérdida)."""
    return "[" + ",".join(repr(float(x)) for x in np.asarray(vector, dtype=np.float32)) + "]"


def load_chunks(
    conn: psycopg.Connection,
    strategy: str,
    chunks: list[dict],
    file_sha256: str,
    vectors: dict[str, np.ndarray] | None = None,
    embeddings_sha256: str | None = None,
) -> int:
    """Sustituye todos los chunks de una estrategia dentro de una transacción (idempotente).
    Si se dan `vectors` (chunk_id -> embedding), deben cubrir todos los chunks."""
    if vectors is not None:
        missing = [c["chunk_id"] for c in chunks if c["chunk_id"] not in vectors]
        if missing:
            raise RuntimeError(f"faltan embeddings de {len(missing)} chunks (p. ej. {missing[0]}); ejecuta `python cli.py embed`")
    placeholders = ", ".join(["%s"] * len(CHUNK_COLUMNS))
    sql = f"INSERT INTO chunks ({', '.join(CHUNK_COLUMNS.values())}, embedding) VALUES ({placeholders}, %s::vector)"
    rows = [tuple(c.get(field) for field in CHUNK_COLUMNS) + (to_vector_literal(vectors[c["chunk_id"]]) if vectors is not None else None,) for c in chunks]
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE strategy = %s", (strategy,))
        cur.executemany(sql, rows)
        cur.execute(
            "INSERT INTO load_info (strategy, chunks_sha256, n_chunks, embeddings_sha256) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (strategy) DO UPDATE SET chunks_sha256 = EXCLUDED.chunks_sha256, n_chunks = EXCLUDED.n_chunks, "
            "embeddings_sha256 = EXCLUDED.embeddings_sha256",
            (strategy, file_sha256, len(rows), embeddings_sha256),
        )
    return len(rows)


def run_init() -> int:
    try:
        with connect() as conn:
            init_db(conn)
    except (RuntimeError, psycopg.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Esquema creado/actualizado.")
    return 0


def run_load() -> int:
    from scripts import embeddings
    from scripts.evaluate import CHUNK_FILES
    from scripts.evaluate import load_chunks as read_chunk_file

    try:
        manifest = embeddings.read_manifest()
        with connect() as conn:
            init_db(conn)
            for strategy, path in CHUNK_FILES.items():
                digest = sha256_bytes(Path(path).read_bytes())
                ids, matrix = embeddings.load_embeddings(f"chunks_{strategy}")  # se niega si están desfasados
                vectors = dict(zip(ids, matrix, strict=True))
                emb_sha = manifest["files"][f"chunks_{strategy}"]["sha256"]
                n = load_chunks(conn, strategy, read_chunk_file(strategy), digest, vectors, emb_sha)
                print(f"{strategy}: {n} chunks con embedding cargados (chunks {digest[:10]}…, embeddings {emb_sha[:10]}…)")
    except (RuntimeError, FileNotFoundError, psycopg.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0
