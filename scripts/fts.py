"""Recuperación por búsqueda de texto completo de Postgres (FTS).

Decisiones fijadas antes de medir (no se ajustan con el golden set):
- Configuración 'english' (corpus y consultas en inglés).
- La consulta es el OR de los términos alfanuméricos (por defecto Postgres exige TODOS: con una pregunta larga
  casi no devolvería nada); el ranking ordena.
- Ranking `ts_rank_cd` con sus valores por defecto (sin pesos ni normalización).
- Desempate por chunk_id, para que dos ejecuciones den exactamente el mismo orden.
"""

import re

from scripts import db
from scripts.metrics import Article

SEARCH_SQL = """
WITH q AS (SELECT to_tsquery('english', %(query)s) AS query)
SELECT c.chunk_id, c.celex, c.article
FROM chunks c, q
WHERE c.strategy = %(strategy)s AND c.tsv @@ q.query {scope}
ORDER BY ts_rank_cd(c.tsv, q.query) DESC, c.chunk_id
LIMIT %(depth)s
"""
PUBLIC_ONLY = "AND c.tenant_id IS NULL"  # sin inquilino (evaluación): solo el corpus público; con inquilino manda RLS


def build_or_query(text: str) -> str | None:
    """'How quickly, must…' -> 'how | quickly | must | …' (sin repetidos). None si no hay términos.
    Solo pasan [a-z0-9]+, así que el texto de la consulta nunca puede inyectar operadores de tsquery."""
    terms = list(dict.fromkeys(re.findall(r"[a-z0-9]+", text.lower())))
    return " | ".join(terms) if terms else None


class FtsIndex:
    def __init__(self, chunking: str, chunks_sha256: str, schema: str | None = None, tenant: str | None = None):
        self.chunking = chunking
        self.schema = schema
        self.tenant = tenant
        with db.connect(schema) as conn:
            row = conn.execute("SELECT chunks_sha256 FROM load_info WHERE strategy = %s", (chunking,)).fetchone()
            version = conn.execute("SELECT version()").fetchone()[0]
        if row is None:
            raise RuntimeError(f"la base de datos no tiene chunks '{chunking}' cargados; ejecuta `python cli.py load-db`")
        if row[0] != chunks_sha256:
            raise RuntimeError(f"los chunks '{chunking}' de la base de datos no coinciden con data/chunks; ejecuta `python cli.py load-db`")
        self.info = {"postgres": version.split(" on ")[0], "regconfig": "english"}

    def search_chunks(self, query: str, depth: int) -> list[tuple[str, Article]]:
        """Los `depth` mejores chunks como (chunk_id, artículo), de mejor a peor."""
        tsquery = build_or_query(query)
        if tsquery is None:
            return []
        with db.connect(self.schema) as conn:
            if self.tenant is not None:
                db.tenant_transaction(conn, self.tenant)  # RLS decide qué filas se ven
            sql = SEARCH_SQL.format(scope="" if self.tenant is not None else PUBLIC_ONLY)
            rows = conn.execute(sql, {"query": tsquery, "strategy": self.chunking, "depth": depth}).fetchall()
        return [(chunk_id, (celex, article)) for chunk_id, celex, article in rows]

    def search(self, query: str, depth: int) -> list[Article]:
        return [article for _, article in self.search_chunks(query, depth)]
