from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_QUERY_CHARS = 500


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"query": "Lead Overseer", "top_k": 10, "mode": "hybrid", "chunking": "window"}})

    query: str = Field(description=f"Consulta en inglés, de 1 a {MAX_QUERY_CHARS} caracteres. Las consultas cortas, como una sigla (TLPT), son válidas.")
    top_k: int = Field(default=10, ge=1, le=50, description="Número de artículos distintos que se devuelven.")
    mode: Literal["hybrid", "fts", "dense"] = Field(
        default="hybrid",
        description=(
            "hybrid = FTS + denso con RRF (k=60); fts = búsqueda de texto completo de Postgres; dense = coseno sobre embeddings. "
            "hybrid y dense necesitan el modelo (requirements-embeddings.txt)."
        ),
    )
    chunking: Literal["window", "article"] = Field(default="window", description="Ventanas de 200 palabras (mejor nDCG medido) o un chunk por artículo.")


class Hit(BaseModel):
    rank: int
    chunk_id: str = Field(description="Chunk que mejor encajó con la consulta dentro de este artículo.")
    act: str
    celex: str
    article: int
    title: str
    snippet: str = Field(description="Fragmento del chunk que mejor encajó, sin la cabecera «Article N. Título».")


class RetrievalResponse(BaseModel):
    query: str
    mode: str
    chunking: str
    top_k: int
    results: list[Hit]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    model: bool
    tenants_enabled: bool
