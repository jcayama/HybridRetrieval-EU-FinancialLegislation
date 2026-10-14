import numpy as np
import psycopg
import pytest

from scripts import bench_index as bi
from scripts import db, embeddings

# ---------- sin base de datos ----------


def unit_rows(n, seed=0):
    m = np.random.default_rng(seed).standard_normal((n, 384)).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_sigma_gives_the_requested_cosine():
    assert bi.sigma_for(1.0) == 0.0
    sigma = bi.sigma_for(0.93)
    assert 1 / np.sqrt(1 + sigma**2) == pytest.approx(0.93)


def test_perturb_hits_the_target_cosine_exactly_and_is_normalized():
    sources = unit_rows(50)
    out = bi.perturb(sources, 0.93, np.random.default_rng(1))
    assert out.dtype == np.float32 and out.shape == sources.shape
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.sum(out * sources, axis=1), 0.93, atol=1e-4)


def test_synthetic_dataset_is_deterministic_and_has_the_right_shape():
    real = unit_rows(30)
    a, qa = bi.synthetic_dataset(real, 500, 7, seed=3)
    b, qb = bi.synthetic_dataset(real, 500, 7, seed=3)
    c, _ = bi.synthetic_dataset(real, 500, 7, seed=4)
    assert a.shape == (500, 384) and qa.shape == (7, 384)
    assert np.array_equal(a, b) and np.array_equal(qa, qb) and not np.array_equal(a, c)
    assert not np.array_equal(a[:7], qa)  # las consultas no son copias de los datos


def test_vector_text_roundtrips_float32_exactly_with_9_digits():
    v = np.random.default_rng(2).standard_normal(384).astype(np.float32)
    back = np.array(bi.vector_text(v).strip("[]").split(","), dtype=np.float64).astype(np.float32)
    assert np.array_equal(back, v)


def test_recall_at_k():
    assert bi.recall_at_k([1, 2, 3], [1, 2, 3]) == 1.0
    assert bi.recall_at_k([3, 2, 1], [1, 2, 3]) == 1.0  # el orden no importa, solo el conjunto
    assert bi.recall_at_k([1, 9, 8], [1, 2, 3]) == pytest.approx(1 / 3)
    assert bi.recall_at_k([1], [1, 2, 3, 4]) == pytest.approx(1 / 4)  # devolver menos de k cuenta como fallo
    assert bi.recall_at_k([], [1, 2]) == 0.0


def test_latency_stats():
    stats = bi.latency_stats([float(x) for x in range(1, 101)])
    assert stats["median_ms"] == pytest.approx(50.5) and stats["p95_ms"] == 95.0
    assert bi.latency_stats([7.0]) == {"median_ms": 7.0, "p95_ms": 7.0}


def test_numpy_topk_matches_brute_force():
    m = unit_rows(40, 5)
    q = unit_rows(3, 6)
    for row, ids in zip(q @ m.T, bi.numpy_topk(m, q, 5), strict=True):
        assert ids == list(np.argsort(-row, kind="stable")[:5])


# ---------- con base de datos (se salta si no hay Postgres) ----------


@pytest.fixture
def bench_conn():
    name = "bench_test_" + __import__("secrets").token_hex(4)
    try:
        conn = psycopg.connect(**db.conninfo(), autocommit=True)
    except (RuntimeError, psycopg.OperationalError) as error:
        pytest.skip(f"Postgres no disponible: {error}")
    with conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
        conn.execute(f"CREATE SCHEMA {name}")
        conn.execute(f"SET search_path = {name}, public")
        try:
            yield conn, name
        finally:
            conn.execute("RESET search_path")
            conn.execute(f"DROP SCHEMA {name} CASCADE")


def test_small_synthetic_benchmark_end_to_end(bench_conn):
    conn, _ = bench_conn
    _, real = embeddings.load_embeddings("chunks_window")
    out = bi.run_synthetic(conn, real, n=3000, n_queries=12, check_variability=True)
    assert out["rows"] == 3000 and out["queries"] == 12
    assert out["exact"]["plan_is_seq_scan"] is True  # la referencia es de verdad un escaneo exacto
    assert out["exact"]["agreement_with_numpy"] >= 0.99  # el ground truth de Postgres coincide con NumPy
    for kind in ("hnsw", "ivfflat"):
        assert out[kind]["size_mb"] > 0 and out[kind]["build_seconds"] >= 0
        assert all(p["uses_index"] for p in out[kind]["sweep"]), f"{kind}: el plan no usa el índice"
        assert all(0.0 <= p["recall@10"] <= 1.0 for p in out[kind]["sweep"])
    assert out["hnsw"]["params"] == {"m": 16, "ef_construction": 64}
    assert out["ivfflat"]["params"] == {"lists": 3}  # filas/1000
    assert out["hnsw"]["sweep"][-1]["recall@10"] >= 0.8  # ef_search=160 sobre 3000 vectores
    assert "rebuild_check" in out["hnsw"]


def test_small_real_corpus_benchmark_reports_golden_effect(bench_conn):
    conn, _ = bench_conn
    out = bi.run_real(conn, "article")
    assert out["rows"] == 248 and out["queries"] == 40
    assert out["ivfflat"]["params"] == {"lists": 16}  # round(sqrt(248))
    exact = out["golden_exact"]["ndcg@10"]
    assert exact == pytest.approx(0.555, abs=0.002)  # coincide con results/dense_article.json (misma búsqueda exacta)
    assert all(0.0 <= g["ndcg@10"] <= 1.0 for g in out["hnsw"]["golden"] + out["ivfflat"]["golden"])


def test_benchmark_removes_its_schema(bench_conn):
    conn, name = bench_conn
    assert conn.execute("SELECT count(*) FROM pg_namespace WHERE nspname = %s", (name,)).fetchone()[0] == 1


def test_scan_switches_force_the_expected_plan_even_when_an_index_exists(bench_conn):
    conn, _ = bench_conn
    matrix = unit_rows(8000, 9)
    conn.execute("CREATE TABLE t (id integer PRIMARY KEY, embedding vector(384) NOT NULL)")
    bi.copy_vectors(conn, "t", matrix)
    conn.execute("ANALYZE t")
    bi.build_index(conn, "t", "hnsw", 8000)
    # con las banderas del planificador por defecto, a esta escala el índice es más barato que el escaneo...
    bi.set_param(conn, "enable_seqscan", "on")
    bi.set_param(conn, "enable_indexscan", "on")
    assert "t_ann_idx" in bi.plan_text(conn, "t", matrix[0])
    # ...y use_exact_scan debe imponer el escaneo exacto igualmente
    bi.use_exact_scan(conn)
    assert "t_ann_idx" not in bi.plan_text(conn, "t", matrix[0])
    bi.use_index_scan(conn)
    assert "t_ann_idx" in bi.plan_text(conn, "t", matrix[0])
