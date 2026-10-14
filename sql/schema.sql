-- Esquema del recuperador. Idempotente: se puede ejecutar varias veces (`python cli.py init-db`).

-- La extensión SIEMPRE en `public`: si se creara dentro de un esquema temporal (tests), al borrarlo con CASCADE
-- se llevaría por delante las columnas de tipo vector de otras tablas.
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    text PRIMARY KEY,
    strategy    text    NOT NULL CHECK (strategy IN ('article', 'window')),
    celex       text    NOT NULL,
    act         text    NOT NULL,
    article     integer NOT NULL,
    title       text    NOT NULL,
    chapter_id  text,
    window_index integer,  -- 'window' es palabra reservada de SQL
    word_start  integer,
    word_end    integer,
    text        text    NOT NULL,
    n_words     integer NOT NULL,
    -- Búsqueda de texto completo: configuración 'english' (el corpus está en inglés).
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_article_idx ON chunks (strategy, celex, article);

-- Qué ficheros de chunks hay cargados (para detectar una base de datos desfasada).
CREATE TABLE IF NOT EXISTS load_info (
    strategy      text PRIMARY KEY,
    chunks_sha256 text    NOT NULL,
    n_chunks      integer NOT NULL
);

-- Búsqueda densa: embedding del chunk (bge-small-en-v1.5, 384 dimensiones, normalizado). Búsqueda exacta por
-- coseno, sin índice aproximado (HNSW/IVFFlat se evalúan aparte).
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding vector(384);
ALTER TABLE load_info ADD COLUMN IF NOT EXISTS embeddings_sha256 text;
