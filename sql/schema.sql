-- Esquema del recuperador. Idempotente: se puede ejecutar varias veces (`python cli.py init-db`).

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
