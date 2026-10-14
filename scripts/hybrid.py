"""Recuperación híbrida: RRF (k = 60) sobre los rankings de FTS y denso.

Decisiones fijadas antes de medir (no se ajustan con el golden set):
- Entradas: FTS de Postgres y búsqueda densa exacta, cada una con sus `depth` mejores chunks.
- Fusión con RRF, k = 60, a nivel de CHUNK (por chunk_id); después se convierte a artículos distintos, igual
  que en los demás sistemas.
- Desempate por chunk_id (lo hace `rrf`).
"""

from scripts import rrf
from scripts.dense import DenseIndex, QueryEmbedder
from scripts.fts import FtsIndex
from scripts.metrics import Article

K = rrf.K_DEFAULT


class RrfIndex:
    """Fusión RRF de un índice léxico y uno denso. Ambos deben exponer `search_chunks(query, depth)`."""

    def __init__(self, lexical, dense, lexical_name: str, info: dict | None = None):
        self.lexical = lexical
        self.dense = dense
        self.info = (getattr(dense, "info", {}) | getattr(lexical, "info", {})) | {"fusion": "RRF", "k": K, "inputs": [lexical_name, "dense"]} | (info or {})

    def search_chunks(self, query: str, depth: int) -> list[tuple[str, Article]]:
        lexical = self.lexical.search_chunks(query, depth)
        semantic = self.dense.search_chunks(query, depth)
        articles = dict(lexical) | dict(semantic)
        fused = rrf.rrf([[chunk_id for chunk_id, _ in lexical], [chunk_id for chunk_id, _ in semantic]], k=K)
        return [(chunk_id, articles[chunk_id]) for chunk_id in fused[:depth]]

    def search(self, query: str, depth: int) -> list[Article]:
        return [article for _, article in self.search_chunks(query, depth)]


class HybridIndex(RrfIndex):
    """El híbrido principal: FTS de Postgres + denso."""

    def __init__(self, chunking: str, chunks_sha256: str, embed_query: QueryEmbedder, schema: str | None = None, tenant: str | None = None):
        fts = FtsIndex(chunking, chunks_sha256, schema, tenant)
        dense = DenseIndex(chunking, chunks_sha256, embed_query, schema, tenant)
        super().__init__(fts, dense, "fts")
        self.fts = fts
