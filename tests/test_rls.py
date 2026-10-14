"""Aislamiento multi-inquilino con Row-Level Security.

Datos: corpus público (tenant_id NULL), documentos privados de A y de B. Todos los datos son SINTÉTICOS.
Los tests se conectan como el propietario (superusuario en desarrollo y CI) y adoptan el rol sin privilegios con
`db.tenant_transaction`, exactamente como hará la aplicación.
"""

import numpy as np
import psycopg
import pytest

from scripts import db, embeddings
from scripts.dense import DenseIndex
from scripts.embeddings import DIM
from scripts.fts import FtsIndex
from scripts.hybrid import HybridIndex

PUBLIC, ART_A, ART_B = 1, 101, 201  # número de artículo de cada fila de ejemplo


def unit(*pairs):
    v = np.zeros(DIM, dtype=np.float32)
    for position, value in pairs:
        v[position] = value
    return v / np.linalg.norm(v)


def row(chunk_id, article, text, vector, tenant=None, strategy="article"):
    return {"chunk_id": chunk_id, "strategy": strategy, "celex": "C", "act": "A", "article": article, "title": "T", "text": text, "n_words": len(text.split()), "vector": vector, "tenant": tenant}


def insert(conn, rows):
    """Inserción como PROPIETARIO (se salta RLS): prepara los datos de la prueba."""
    sql = (
        "INSERT INTO chunks (chunk_id, strategy, celex, act, article, title, text, n_words, embedding, tenant_id) "
        "VALUES (%(chunk_id)s, %(strategy)s, %(celex)s, %(act)s, %(article)s, %(title)s, %(text)s, %(n_words)s, %(embedding)s::vector, %(tenant)s)"
    )
    with conn.cursor() as cur:
        cur.executemany(sql, [{**r, "celex": "C", "act": "A", "title": "T", "embedding": db.to_vector_literal(r["vector"])} for r in rows])


def seed(schema, extra=()):
    rows = [
        row("pub1", PUBLIC, "alpha public regulation text", unit((0, 1))),
        row("a1", ART_A, "alpha CANARYA internal policy of tenant A", unit((1, 1))),
        row("b1", ART_B, "alpha CANARYB internal policy of tenant B", unit((2, 1)), tenant="B"),
        *extra,
    ]
    rows[1]["tenant"] = "A"
    with db.connect(schema) as conn:
        insert(conn, rows)
        sha = embeddings.read_manifest()["files"]["chunks_article"]["sha256"]
        conn.execute("INSERT INTO load_info (strategy, chunks_sha256, n_chunks, embeddings_sha256) VALUES ('article', 'c', 3, %s) ON CONFLICT (strategy) DO NOTHING", (sha,))


def as_tenant(schema, tenant, sql, params=()):
    with db.connect(schema) as conn:
        db.tenant_transaction(conn, tenant)
        return conn.execute(sql, params).fetchall()


# ---------- el rol y la tabla ----------


def test_app_role_is_unprivileged_and_the_table_forces_rls(schema):
    with db.connect(schema) as conn:
        assert conn.execute("SELECT rolsuper, rolbypassrls, rolcanlogin, rolcreaterole FROM pg_roles WHERE rolname = 'app_user'").fetchone() == (False, False, False, False)
        owner, enabled, forced = conn.execute("SELECT relowner::regrole::text, relrowsecurity, relforcerowsecurity FROM pg_class WHERE oid = 'chunks'::regclass").fetchone()
        assert owner != "app_user" and enabled is True and forced is True
        assert {r[0] for r in conn.execute("SELECT polname FROM pg_policy WHERE polrelid = 'chunks'::regclass").fetchall()} == {"chunks_select", "chunks_insert", "chunks_update", "chunks_delete"}


# ---------- lectura ----------


def test_each_tenant_sees_the_public_corpus_plus_its_own_rows_and_never_the_others(schema):
    seed(schema)
    assert as_tenant(schema, "A", "SELECT chunk_id FROM chunks ORDER BY chunk_id") == [("a1",), ("pub1",)]
    assert as_tenant(schema, "B", "SELECT chunk_id FROM chunks ORDER BY chunk_id") == [("b1",), ("pub1",)]
    assert as_tenant(schema, "C-nobody", "SELECT chunk_id FROM chunks") == [("pub1",)]


def test_no_tenant_means_no_rows_not_even_public_ones(schema):
    seed(schema)
    with db.connect(schema) as conn:
        conn.execute("SET LOCAL ROLE app_user")  # rol sin privilegios pero SIN inquilino fijado
        assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
        conn.execute("SELECT set_config('app.tenant_id', '', true)")
        assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0  # inquilino vacío = no fijado


def test_the_owner_bypasses_rls_which_is_why_the_role_switch_is_needed(schema):
    seed(schema)
    with db.connect(schema) as conn:
        if not conn.execute("SELECT rolsuper FROM pg_roles WHERE rolname = current_user").fetchone()[0]:
            pytest.skip("la conexión de prueba no es superusuaria")
        assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 3  # sin cambiar de rol se ve TODO


# ---------- conexiones reutilizadas (pool) ----------


def test_tenant_and_role_do_not_survive_the_transaction(schema):
    seed(schema)
    with db.connect(schema) as conn:
        start_role = conn.execute("SELECT current_user").fetchone()[0]
        db.tenant_transaction(conn, "A")
        assert conn.execute("SELECT current_user, current_setting('app.tenant_id')").fetchone() == ("app_user", "A")
        conn.commit()  # fin de la «petición»: la conexión vuelve al pool
        assert conn.execute("SELECT current_user").fetchone()[0] == start_role  # el rol no se arrastra
        assert conn.execute("SELECT current_setting('app.tenant_id', true)").fetchone()[0] in (None, "")  # el inquilino tampoco


def test_negative_control_a_session_level_setting_does_leak_across_requests(schema):
    """Demuestra que el fallo existe con un SET de sesión: así se ve que el test anterior detecta la fuga."""
    seed(schema)
    with db.connect(schema) as conn:
        conn.execute("SELECT set_config('app.tenant_id', 'A', false)")  # is_local = false: el error clásico
        conn.commit()
        assert conn.execute("SELECT current_setting('app.tenant_id', true)").fetchone()[0] == "A"  # sobrevive a la transacción


def test_tenant_transaction_validates_its_input(schema):
    with db.connect(schema) as conn:
        for bad in ("", "   ", None, 7):
            with pytest.raises(ValueError):
                db.tenant_transaction(conn, bad)
    with psycopg.connect(**db.conninfo(), autocommit=True) as conn, pytest.raises(RuntimeError, match="autocommit"):
        db.tenant_transaction(conn, "A")


# ---------- escritura ----------


def test_writes_are_confined_to_the_tenants_own_rows(schema):
    seed(schema)
    new = row("a2", 102, "new private row", unit((3, 1)), tenant="A")
    with db.connect(schema) as conn:
        db.tenant_transaction(conn, "A")
        insert(conn, [new])  # propia: permitido
        for forbidden in (row("x1", 900, "as B", unit((4, 1)), tenant="B"), row("x2", 901, "as public", unit((5, 1)), tenant=None)):
            with pytest.raises(psycopg.errors.InsufficientPrivilege), conn.transaction():
                insert(conn, [forbidden])
        assert conn.execute("UPDATE chunks SET text = 'hacked' WHERE chunk_id = 'pub1'").rowcount == 0  # no puede tocar el corpus público
        assert conn.execute("DELETE FROM chunks WHERE chunk_id = 'pub1'").rowcount == 0
        assert conn.execute("UPDATE chunks SET text = 'hacked' WHERE chunk_id = 'b1'").rowcount == 0  # ni las filas de otro inquilino
        assert conn.execute("DELETE FROM chunks WHERE chunk_id = 'b1'").rowcount == 0
        assert conn.execute("UPDATE chunks SET text = 'edited' WHERE chunk_id = 'a1'").rowcount == 1  # sí las suyas
        assert conn.execute("DELETE FROM chunks WHERE chunk_id = 'a2'").rowcount == 1
    with db.connect(schema) as conn:  # lo que quedó, visto por el propietario
        rows = dict(conn.execute("SELECT chunk_id, text FROM chunks").fetchall())
    assert rows["pub1"] == "alpha public regulation text" and rows["b1"].startswith("alpha CANARYB") and rows["a1"] == "edited" and "a2" not in rows


def test_a_tenant_cannot_reassign_its_rows_to_another_tenant(schema):
    seed(schema)
    with db.connect(schema) as conn:
        db.tenant_transaction(conn, "A")
        with pytest.raises(psycopg.errors.InsufficientPrivilege), conn.transaction():
            conn.execute("UPDATE chunks SET tenant_id = 'B' WHERE chunk_id = 'a1'")  # WITH CHECK de UPDATE
        with pytest.raises(psycopg.errors.InsufficientPrivilege), conn.transaction():
            conn.execute("UPDATE chunks SET tenant_id = NULL WHERE chunk_id = 'a1'")  # tampoco convertirlas en públicas


# ---------- búsquedas ----------


def articles(results):
    return {article for _, article in results}


def test_fts_search_never_returns_another_tenants_rows(schema):
    seed(schema)
    for tenant, own, other in (("A", ART_A, ART_B), ("B", ART_B, ART_A)):
        found = articles(FtsIndex("article", "c", schema, tenant=tenant).search("alpha", 10))
        assert found == {PUBLIC, own} and other not in found
    assert FtsIndex("article", "c", schema, tenant="B").search("CANARYA", 10) == []  # el testigo de A es invisible para B
    assert articles(FtsIndex("article", "c", schema, tenant=None).search("alpha", 10)) == {PUBLIC}  # evaluación: solo el corpus público


def test_dense_search_never_returns_another_tenants_rows(schema):
    seed(schema)
    near_a = unit((1, 1))  # idéntico al vector privado de A
    as_b = articles(DenseIndex("article", "c", lambda q: near_a, schema, tenant="B").search("q", 10))
    as_a = articles(DenseIndex("article", "c", lambda q: near_a, schema, tenant="A").search("q", 10))
    public_only = articles(DenseIndex("article", "c", lambda q: near_a, schema, tenant=None).search("q", 10))
    assert ART_A not in as_b and as_b == {PUBLIC, ART_B}
    assert ART_A in as_a and ART_B not in as_a
    assert public_only == {PUBLIC}


def test_hybrid_search_never_returns_another_tenants_rows(schema):
    seed(schema)
    found = articles(HybridIndex("article", "c", lambda q: unit((1, 1)), schema, tenant="B").search("alpha CANARYA", 10))
    assert ART_A not in found and found <= {PUBLIC, ART_B}


# ---------- índice aproximado + RLS ----------


@pytest.fixture
def crowded_neighbourhood(schema):
    """3.000 filas públicas lejos de la consulta y 200 de A pegadas a ella; B no tiene filas cerca. HNSW sobre todo."""
    rng = np.random.default_rng(0)
    query = unit((0, 1))
    public = rng.standard_normal((3000, DIM)).astype(np.float32)
    public[:, 0] = 0.0  # ortogonales a la consulta
    public /= np.linalg.norm(public, axis=1, keepdims=True)
    rows = [row(f"p{i:04d}", i + 1, "public", v) for i, v in enumerate(public)]
    rows += [row(f"a{i:03d}", 10_000 + i, "private A", unit((0, 1), (1 + i % 300, 0.05)), tenant="A") for i in range(200)]
    with db.connect(schema) as conn:
        try:
            conn.execute("SELECT set_config('hnsw.iterative_scan', 'off', true)")
        except psycopg.Error:
            pytest.skip("esta versión de pgvector no tiene búsquedas iterativas")
        insert(conn, rows)
        conn.execute("CREATE INDEX chunks_hnsw ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)")
        conn.execute("ANALYZE chunks")
    return schema, query


KNN = "SELECT chunk_id, tenant_id FROM chunks ORDER BY embedding <=> %s::vector LIMIT 10"


def knn_as(schema, tenant, query, settings):
    with db.connect(schema) as conn:
        db.tenant_transaction(conn, tenant)
        for name, value in settings.items():
            conn.execute("SELECT set_config(%s, %s, true)", (name, value))
        return conn.execute(KNN, (db.to_vector_literal(query),)).fetchall()


FORCE_INDEX = {"enable_seqscan": "off"}


def test_hnsw_plus_rls_never_leaks_but_returns_fewer_rows_than_asked(crowded_neighbourhood):
    schema, query = crowded_neighbourhood
    results = knn_as(schema, "B", query, FORCE_INDEX | {"hnsw.ef_search": "40"})
    assert all(tenant is None for _, tenant in results)  # jamás una fila de A
    assert len(results) < 10  # LIMITACIÓN conocida: HNSW devuelve 40 candidatos, todos de A, y RLS los descarta


def test_iterative_scan_restores_the_requested_number_of_rows_without_leaking(crowded_neighbourhood):
    schema, query = crowded_neighbourhood
    results = knn_as(schema, "B", query, FORCE_INDEX | {"hnsw.ef_search": "40", "hnsw.iterative_scan": "relaxed_order"})
    assert len(results) == 10 and all(tenant is None for _, tenant in results)


def test_exact_scan_with_rls_returns_the_full_page_and_never_leaks(crowded_neighbourhood):
    schema, query = crowded_neighbourhood
    results = knn_as(schema, "B", query, {"enable_indexscan": "off", "enable_bitmapscan": "off"})
    assert len(results) == 10 and all(tenant is None for _, tenant in results)
    own = knn_as(schema, "A", query, {"enable_indexscan": "off", "enable_bitmapscan": "off"})
    assert all(tenant == "A" for _, tenant in own)  # A ve primero sus 200 filas cercanas
