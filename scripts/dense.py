"""Recuperación densa: similitud coseno exacta sobre los embeddings guardados en Postgres (pgvector).

Decisiones fijadas antes de medir (no se ajustan con el golden set):
- Distancia coseno (`<=>`) sobre vectores normalizados, con BÚSQUEDA EXACTA (sin índice aproximado).
- Desempate por chunk_id, para que dos ejecuciones den exactamente el mismo orden.
- El vector de la consulta lo aporta un «embebedor» inyectable. Para el golden set se usan los embeddings
  precalculados (sin torch); una API futura podrá inyectar uno que use el modelo.
"""

from collections.abc import Callable

import numpy as np

from scripts import db, embeddings
from scripts.golden import QUERIES_FILE, load_jsonl
from scripts.metrics import Article

SEARCH_SQL = """
SELECT chunk_id, celex, article
FROM chunks
WHERE strategy = %(strategy)s AND embedding IS NOT NULL
ORDER BY embedding <=> %(vector)s::vector, chunk_id
LIMIT %(depth)s
"""

QueryEmbedder = Callable[[str], np.ndarray]


def precomputed_query_embedder() -> QueryEmbedder:
    """Embebedor que solo conoce las consultas del golden set (vectores precalculados y verificados)."""
    ids, matrix = embeddings.load_embeddings("queries")
    by_id = dict(zip(ids, matrix, strict=True))
    by_text = {row["query"]: by_id[row["query_id"]] for row in load_jsonl(QUERIES_FILE)}

    def embed(query: str) -> np.ndarray:
        if query not in by_text:
            raise KeyError(f"la consulta no tiene embedding precalculado: {query!r}")
        return by_text[query]

    return embed


class DenseIndex:
    def __init__(self, chunking: str, chunks_sha256: str, embed_query: QueryEmbedder, schema: str | None = None):
        self.chunking = chunking
        self.embed_query = embed_query
        self.schema = schema
        with db.connect(schema) as conn:
            row = conn.execute("SELECT chunks_sha256, embeddings_sha256 FROM load_info WHERE strategy = %s", (chunking,)).fetchone()
            version = conn.execute("SELECT version()").fetchone()[0]
            pgvector = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
        if row is None:
            raise RuntimeError(f"la base de datos no tiene chunks '{chunking}' cargados; ejecuta `python cli.py load-db`")
        if row[0] != chunks_sha256:
            raise RuntimeError(f"los chunks '{chunking}' de la base de datos no coinciden con data/chunks; ejecuta `python cli.py load-db`")
        expected = embeddings.read_manifest()["files"][f"chunks_{chunking}"]["sha256"]
        if row[1] != expected:
            raise RuntimeError(f"los embeddings '{chunking}' de la base de datos no coinciden con data/embeddings; ejecuta `python cli.py load-db`")
        self.info = {
            "postgres": version.split(" on ")[0],
            "pgvector": pgvector[0] if pgvector else None,
            "model": embeddings.MODEL_ID,
            "revision": embeddings.REVISION,
            "embeddings_sha256": expected,
        }

    def search_chunks(self, query: str, depth: int) -> list[tuple[str, Article]]:
        """Los `depth` mejores chunks como (chunk_id, artículo), de mejor a peor."""
        vector = db.to_vector_literal(self.embed_query(query))
        with db.connect(self.schema) as conn:
            rows = conn.execute(SEARCH_SQL, {"strategy": self.chunking, "vector": vector, "depth": depth}).fetchall()
        return [(chunk_id, (celex, article)) for chunk_id, celex, article in rows]

    def search(self, query: str, depth: int) -> list[Article]:
        return [article for _, article in self.search_chunks(query, depth)]
