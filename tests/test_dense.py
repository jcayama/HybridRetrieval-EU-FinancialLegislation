import numpy as np
import pytest

from scripts import db, embeddings
from scripts.dense import DenseIndex, precomputed_query_embedder
from scripts.embeddings import DIM


def unit(*pairs):
    """Vector normalizado de DIM dimensiones con los pesos dados {posición: valor}."""
    v = np.zeros(DIM, dtype=np.float32)
    for position, value in pairs:
        v[position] = value
    return v / np.linalg.norm(v)


def chunk(strategy, article, chunk_id=None, celex="C"):
    return {
        "chunk_id": chunk_id or f"{celex}:art_{article}:{strategy}",
        "strategy": strategy,
        "celex": celex,
        "act": "A",
        "article": article,
        "title": "T",
        "chapter_id": None,
        "text": "text",
        "n_words": 1,
    }


def load(schema, strategy, chunks, vectors, sha="c", emb_sha=None):
    emb_sha = emb_sha or embeddings.read_manifest()["files"][f"chunks_{strategy}"]["sha256"]
    with db.connect(schema) as conn:
        db.load_chunks(conn, strategy, chunks, sha, vectors, emb_sha)


def index(schema, strategy, vector, sha="c"):
    return DenseIndex(strategy, sha, lambda query: vector, schema)


# ---------- sin base de datos ----------


def test_vector_literal_is_exact_for_float32():
    rng = np.random.default_rng(0)
    v = rng.normal(size=DIM).astype(np.float32)
    literal = db.to_vector_literal(v)
    assert literal.startswith("[") and literal.endswith("]")
    assert np.array_equal(np.array(literal[1:-1].split(","), dtype=np.float64).astype(np.float32), v)


def test_precomputed_embedder_knows_golden_queries_only():
    embed = precomputed_query_embedder()
    vector = embed("Lead Overseer")
    assert vector.shape == (DIM,) and np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-3)
    with pytest.raises(KeyError, match="precalculado"):
        embed("una consulta que no está en el golden set")


# ---------- con base de datos (se saltan si no hay Postgres) ----------


def test_pgvector_extension_lives_in_public_so_dropping_test_schemas_is_safe(schema):
    with db.connect(schema) as conn:
        assert conn.execute("SELECT extnamespace::regnamespace::text FROM pg_extension WHERE extname = 'vector'").fetchone()[0] == "public"


def test_vectors_roundtrip_through_postgres_exactly(schema):
    v = np.random.default_rng(1).normal(size=DIM).astype(np.float32)
    load(schema, "article", [chunk("article", 1)], {"C:art_1:article": v})
    with db.connect(schema) as conn:
        text = conn.execute("SELECT embedding::text FROM chunks").fetchone()[0]
    assert np.array_equal(np.array(text.strip("[]").split(","), dtype=np.float64).astype(np.float32), v)


def test_search_orders_by_cosine_similarity(schema):
    chunks = [chunk("article", 1), chunk("article", 2), chunk("article", 3)]
    vectors = {"C:art_1:article": unit((0, 1)), "C:art_2:article": unit((1, 1)), "C:art_3:article": unit((1, 1), (0, 0.2))}
    load(schema, "article", chunks, vectors)
    assert index(schema, "article", unit((1, 1), (0, 0.1))).search("q", 10) == [("C", 3), ("C", 2), ("C", 1)]


def test_sql_ranking_matches_numpy_cosine(schema):
    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(8, DIM)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    query = rng.normal(size=DIM).astype(np.float32)
    query /= np.linalg.norm(query)
    chunks = [chunk("article", i + 1) for i in range(8)]
    load(schema, "article", chunks, {c["chunk_id"]: matrix[i] for i, c in enumerate(chunks)})
    expected = [("C", int(i) + 1) for i in np.argsort(-(matrix @ query))]
    assert index(schema, "article", query).search("q", 8) == expected


def test_ties_are_broken_by_chunk_id(schema):
    # 40 chunks con vectores idénticos y ids insertados en orden desordenado: sin desempate explícito, Postgres
    # no garantiza ningún orden entre ellos
    same = unit((0, 1))
    numbers = [(i * 17) % 40 for i in range(40)]
    chunks = [chunk("article", n + 1, chunk_id=f"id{n:02d}") for n in numbers]
    load(schema, "article", chunks, {c["chunk_id"]: same for c in chunks})
    result = index(schema, "article", same).search("q", 10)
    assert result == [("C", n + 1) for n in range(10)]  # id00..id09, en orden


def test_search_filters_strategy_respects_depth_and_allows_repeated_articles(schema):
    load(schema, "article", [chunk("article", 1)], {"C:art_1:article": unit((0, 1))})
    windows = [chunk("window", 7, chunk_id="w0"), chunk("window", 7, chunk_id="w1")]
    load(schema, "window", windows, {"w0": unit((0, 1)), "w1": unit((0, 1), (1, 0.1))})
    assert index(schema, "article", unit((0, 1))).search("q", 10) == [("C", 1)]
    dense_windows = index(schema, "window", unit((0, 1)))
    assert dense_windows.search("q", 10) == [("C", 7), ("C", 7)]
    assert len(dense_windows.search("q", 1)) == 1


def test_chunks_without_embedding_are_excluded(schema):
    chunks = [chunk("article", 1), chunk("article", 2)]
    with db.connect(schema) as conn:
        db.load_chunks(conn, "article", chunks, "c", None, None)  # sin vectores: embedding NULL
    with db.connect(schema) as conn:
        conn.execute("UPDATE load_info SET embeddings_sha256 = %s", (embeddings.read_manifest()["files"]["chunks_article"]["sha256"],))
    assert index(schema, "article", unit((0, 1))).search("q", 10) == []


def test_load_requires_a_vector_for_every_chunk(schema):
    with pytest.raises(RuntimeError, match="faltan embeddings"):
        load(schema, "article", [chunk("article", 1), chunk("article", 2)], {"C:art_1:article": unit((0, 1))})


def test_stale_database_is_detected(schema):
    with pytest.raises(RuntimeError, match="load-db"):
        index(schema, "article", unit((0, 1)))  # nada cargado
    load(schema, "article", [chunk("article", 1)], {"C:art_1:article": unit((0, 1))}, sha="old")
    with pytest.raises(RuntimeError, match="chunks"):
        index(schema, "article", unit((0, 1)), sha="new")
    load(schema, "article", [chunk("article", 1)], {"C:art_1:article": unit((0, 1))}, emb_sha="otro-hash")
    with pytest.raises(RuntimeError, match="embeddings"):
        index(schema, "article", unit((0, 1)))


def test_search_scored_returns_the_cosine_in_the_same_order_as_search_chunks(schema):
    near, far = unit((0, 1), (1, 1)), unit((1, 1))  # coseno con la consulta (0,1): 1/√2 y 0
    chunks = [chunk("article", 1), chunk("article", 2), chunk("article", 3)]
    load(schema, "article", chunks, {"C:art_1:article": unit((0, 1)), "C:art_2:article": near, "C:art_3:article": far})
    idx = index(schema, "article", unit((0, 1)))
    scored = idx.search_scored("q", 10)
    assert [(c, a) for c, a, _ in scored] == idx.search_chunks("q", 10)  # mismo ranking
    assert [cosine for *_, cosine in scored] == pytest.approx([1.0, 1 / np.sqrt(2), 0.0], abs=1e-6)  # el coseno, a mano
    assert [c for *_, c in scored] == sorted((c for *_, c in scored), reverse=True)  # decreciente: el primero es el mejor
