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

-- =====================================================================================================
-- Multi-inquilino con Row-Level Security.
--   tenant_id NULL  = corpus público (visible para todos los inquilinos)
--   tenant_id <id>  = documento privado de ese inquilino
-- La aplicación fija el inquilino con `db.tenant_transaction` (SET LOCAL ROLE app_user + set_config local).
-- =====================================================================================================
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tenant_id text;
CREATE INDEX IF NOT EXISTS chunks_tenant_idx ON chunks (tenant_id);

-- Rol sin privilegios: no es superusuario, no tiene BYPASSRLS y no es propietario de la tabla.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_user') THEN
        CREATE ROLE app_user NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS NOINHERIT;
    END IF;
    EXECUTE format('GRANT app_user TO %I', current_user);  -- para poder hacer SET ROLE aunque el propietario no sea superusuario
    EXECUTE format('GRANT USAGE ON SCHEMA %I TO app_user', current_schema());
END $$;
GRANT SELECT, INSERT, UPDATE, DELETE ON chunks TO app_user;

-- Inquilino de la transacción actual; NULL si no está fijado o está vacío.
CREATE OR REPLACE FUNCTION current_tenant() RETURNS text LANGUAGE sql STABLE
AS $$ SELECT NULLIF(current_setting('app.tenant_id', true), '') $$;

ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks FORCE ROW LEVEL SECURITY;  -- también para el propietario (los superusuarios siempre se la saltan)

DROP POLICY IF EXISTS chunks_select ON chunks;
CREATE POLICY chunks_select ON chunks FOR SELECT
    USING (current_tenant() IS NOT NULL AND (tenant_id IS NULL OR tenant_id = current_tenant()));

DROP POLICY IF EXISTS chunks_insert ON chunks;
CREATE POLICY chunks_insert ON chunks FOR INSERT
    WITH CHECK (current_tenant() IS NOT NULL AND tenant_id = current_tenant());

DROP POLICY IF EXISTS chunks_update ON chunks;
CREATE POLICY chunks_update ON chunks FOR UPDATE
    USING (tenant_id = current_tenant()) WITH CHECK (tenant_id = current_tenant());

DROP POLICY IF EXISTS chunks_delete ON chunks;
CREATE POLICY chunks_delete ON chunks FOR DELETE
    USING (tenant_id = current_tenant());
