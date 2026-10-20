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
    chunking: Literal["window", "article"] = Field(
        default="window",
        description=(
            "window: ventanas de 200 palabras (mejor nDCG medido); el fragmento es el inicio de la ventana que mejor encajó. "
            "article: un chunk por artículo; el fragmento es el pasaje del artículo que más coincide con la consulta."
        ),
    )


class Hit(BaseModel):
    rank: int
    chunk_id: str = Field(description="Chunk que mejor encajó con la consulta dentro de este artículo (`…:w200:N` es una ventana; `…:art_N`, el artículo entero).")
    act: str
    celex: str
    article: int
    title: str
    snippet: str = Field(
        description=(
            "Fragmento de unos 400 caracteres, sin la cabecera «Article N. Título». Con chunking=window es el inicio del chunk; "
            "con chunking=article es el pasaje que más coincide con las palabras de la consulta (ts_headline de Postgres), "
            "con «…» donde se recorta. Si la consulta no comparte palabras con el artículo (p. ej. en modo dense), es el inicio del artículo."
        )
    )
    chunk_words: int = Field(description="Palabras del cuerpo del chunk devuelto: unas 200 con chunking=window; el artículo entero con chunking=article.")


class RetrievalResponse(BaseModel):
    query: str
    mode: str
    chunking: str
    top_k: int
    results: list[Hit]
    abstained: bool = Field(
        default=False,
        description=(
            "true si el servicio no devolvió resultados porque el coseno del mejor chunk quedó bajo el umbral: la consulta parece ajena al corpus. "
            "Solo en los modos dense y hybrid. El umbral solo descarta consultas claramente ajenas; no detecta las «casi del dominio»."
        ),
    )
    best_cosine: float | None = Field(default=None, description="Coseno del mejor chunk de la búsqueda densa (solo en dense y hybrid); null en fts.")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    model: bool
    tenants_enabled: bool
    abstention: bool
