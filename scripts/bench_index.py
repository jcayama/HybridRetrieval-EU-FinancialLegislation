"""Benchmark de índices aproximados de pgvector: HNSW frente a IVFFlat (recall y latencia).

Parámetros fijados antes de medir (no se ajustan con los resultados):
- HNSW: m=16, ef_construction=64 (valores por defecto de pgvector); barrido de ef_search.
- IVFFlat: lists = filas/1000 hasta 1 M de filas (guía de pgvector); para el corpus real, donde eso sale < 1,
  lists = round(sqrt(filas)). Barrido de probes.
- Ground truth: la búsqueda EXACTA de Postgres (escaneo secuencial) sobre los mismos datos; NumPy la contrasta.
- Datos sintéticos: perturbaciones de vectores reales con un coseno fijo respecto a su origen (COSINE_TO_SOURCE).
  Son datos SINTÉTICOS con la estructura de los reales; indican el comportamiento de los índices, no lo garantizan.

Todo ocurre en un esquema temporal que se borra al terminar; no toca las tablas del proyecto.
"""

import json
import math
import platform
import secrets
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import psycopg

from scripts import db, embeddings
from scripts.dense import precomputed_query_embedder
from scripts.download_corpus import ROOT
from scripts.golden import QUERIES_FILE, load_jsonl

RESULTS_FILE = ROOT / "results" / "index_benchmark.json"
DIM = embeddings.DIM
K = 10
HNSW_M, HNSW_EF_CONSTRUCTION = 16, 64
EF_SEARCH = (10, 20, 40, 80, 160)
IVF_PROBES = (1, 2, 5, 10, 20, 50)
SYNTHETIC_SIZE = 100_000
SYNTHETIC_QUERIES = 200
COSINE_TO_SOURCE = 0.93  # ≈ sqrt(0,863): la media del coseno a los 10 vecinos más cercanos en las ventanas reales es 0,863
SEED = 2026
PASSES = 3  # pasadas medidas (hay además una de calentamiento)
MAINTENANCE_WORK_MEM = "1GB"
# 0: sin trabajadores paralelos. En contenedores Docker (/dev/shm = 64 MB por defecto) una construcción paralela con
# maintenance_work_mem alto falla con «could not resize shared memory segment»; además un solo hilo no depende de la
# planificación entre hilos, lo que ayuda a que el grafo sea reproducible.
PARALLEL_MAINTENANCE_WORKERS = 0
DEPTH_END_TO_END = 200  # chunks que pide el sistema denso antes de convertirlos a artículos


# ---------- datos sintéticos ----------


def sigma_for(cosine: float) -> float:
    """Norma del ruido ortogonal que da exactamente ese coseno respecto al vector original."""
    return math.sqrt(1.0 / cosine**2 - 1.0)


def perturb(sources: np.ndarray, cosine: float, rng: np.random.Generator) -> np.ndarray:
    """Vectores normalizados cuyo coseno con cada fuente es `cosine`: fuente + ruido ortogonal de norma sigma."""
    sources64 = sources.astype(np.float64)
    noise = rng.standard_normal(sources64.shape)
    noise -= (noise * sources64).sum(axis=1, keepdims=True) * sources64  # ortogonal a la fuente
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    out = sources64 + sigma_for(cosine) * noise
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out.astype(np.float32)


def synthetic_dataset(real: np.ndarray, n: int, n_queries: int, cosine: float = COSINE_TO_SOURCE, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    parts = []
    for start in range(0, n, 20_000):  # por bloques, para no multiplicar la memoria
        size = min(20_000, n - start)
        parts.append(perturb(real[rng.integers(0, len(real), size=size)], cosine, rng))
    queries = perturb(real[rng.integers(0, len(real), size=n_queries)], cosine, np.random.default_rng(seed + 1))
    return np.concatenate(parts), queries


# ---------- métricas ----------


def recall_at_k(approx: list[int], exact: list[int], k: int = K) -> float:
    """Fracción de los k vecinos exactos que el índice devuelve (si devuelve menos de k, los que faltan son fallos)."""
    return len(set(approx[:k]) & set(exact[:k])) / len(exact[:k])


def latency_stats(times_ms: list[float]) -> dict[str, float]:
    ordered = sorted(times_ms)
    return {"median_ms": statistics.median(ordered), "p95_ms": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]}


def mean_ci(values: list[float]) -> list[float]:
    from scripts.evaluate import mean_ci as bootstrap_mean_ci

    _, low, high = bootstrap_mean_ci({str(i): v for i, v in enumerate(values)})
    return [low, high]


def numpy_topk(matrix: np.ndarray, queries: np.ndarray, k: int = K) -> list[list[int]]:
    scores = queries @ matrix.T
    return [list(row[:k]) for row in np.argsort(-scores, axis=1, kind="stable")]


# ---------- base de datos ----------


def vector_text(vector) -> str:
    """'%.9g' basta para que un float32 viaje exacto (9 dígitos significativos) y es mucho más corto que repr."""
    return "[" + ",".join("%.9g" % x for x in np.asarray(vector, dtype=np.float32).tolist()) + "]"


def set_param(conn, name: str, value) -> None:
    conn.execute("SELECT set_config(%s, %s, false)", (name, str(value)))


def copy_vectors(conn, table: str, matrix: np.ndarray, extra: list[tuple] | None = None) -> None:
    """Carga masiva por COPY. `extra` aporta columnas adicionales (celex, article) por fila."""
    columns = "id, embedding" if extra is None else "id, celex, article, embedding"
    with conn.cursor() as cur, cur.copy(f"COPY {table} ({columns}) FROM STDIN") as copy:
        for i, vector in enumerate(matrix):
            row = (i, vector_text(vector)) if extra is None else (i, *extra[i], vector_text(vector))
            copy.write_row(row)


def knn_ids(conn, table: str, vector: np.ndarray, limit: int = K) -> list[int]:
    sql = f"SELECT id FROM {table} ORDER BY embedding <=> %s::vector LIMIT {int(limit)}"
    return [r[0] for r in conn.execute(sql, (vector_text(vector),)).fetchall()]


def plan_text(conn, table: str, vector: np.ndarray) -> str:
    sql = f"EXPLAIN SELECT id FROM {table} ORDER BY embedding <=> %s::vector LIMIT {K}"
    return "\n".join(r[0] for r in conn.execute(sql, (vector_text(vector),)).fetchall())


def use_exact_scan(conn) -> None:
    set_param(conn, "enable_indexscan", "off")
    set_param(conn, "enable_bitmapscan", "off")
    set_param(conn, "enable_seqscan", "on")


def use_index_scan(conn) -> None:
    set_param(conn, "enable_indexscan", "on")
    set_param(conn, "enable_bitmapscan", "on")
    set_param(conn, "enable_seqscan", "off")  # fuerza el índice: a esta escala el planificador preferiría el escaneo


def timed_queries(conn, table: str, queries: np.ndarray, limit: int = K) -> tuple[list[list[int]], dict[str, float]]:
    """Resultados de la última pasada y latencia: mediana por consulta de PASSES pasadas, tras una de calentamiento."""
    for vector in queries:  # calentamiento
        knn_ids(conn, table, vector, limit)
    per_query = [[] for _ in queries]
    results: list[list[int]] = []
    for _ in range(PASSES):
        results = []
        for i, vector in enumerate(queries):
            start = time.perf_counter()
            ids = knn_ids(conn, table, vector, limit)
            per_query[i].append((time.perf_counter() - start) * 1000)
            results.append(ids)
    return results, latency_stats([statistics.median(t) for t in per_query])


def build_index(conn, table: str, kind: str, rows: int, ivf_lists: int | None = None) -> dict:
    set_param(conn, "maintenance_work_mem", MAINTENANCE_WORK_MEM)
    set_param(conn, "max_parallel_maintenance_workers", PARALLEL_MAINTENANCE_WORKERS)
    conn.execute(f"DROP INDEX IF EXISTS {table}_ann_idx")
    if kind == "hnsw":
        options = f"m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION}"
        params = {"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION}
    else:
        lists = ivf_lists or max(1, rows // 1000)
        options = f"lists = {lists}"
        params = {"lists": lists}
    start = time.perf_counter()
    conn.execute(f"CREATE INDEX {table}_ann_idx ON {table} USING {kind} (embedding vector_cosine_ops) WITH ({options})")
    seconds = time.perf_counter() - start
    size = conn.execute(f"SELECT pg_relation_size('{table}_ann_idx')").fetchone()[0]
    return {"kind": kind, "params": params, "build_seconds": seconds, "size_mb": size / 1e6}


def sweep(conn, table: str, queries: np.ndarray, exact: list[list[int]], kind: str, values: tuple[int, ...]) -> list[dict]:
    setting = "hnsw.ef_search" if kind == "hnsw" else "ivfflat.probes"
    use_index_scan(conn)
    out = []
    for value in values:
        set_param(conn, setting, value)
        plan = plan_text(conn, table, queries[0])
        results, stats = timed_queries(conn, table, queries)
        recalls = [recall_at_k(r, e) for r, e in zip(results, exact, strict=True)]
        out.append(
            {
                setting: value,
                "recall@10": statistics.mean(recalls),
                "recall@10_ci95": mean_ci(recalls),
                "queries_with_fewer_than_10_results": sum(len(r) < K for r in results),
                "uses_index": f"{table}_ann_idx" in plan,
                **stats,
            }
        )
    return out


def exact_baseline(conn, table: str, queries: np.ndarray) -> tuple[list[list[int]], dict]:
    use_exact_scan(conn)
    plan = plan_text(conn, table, queries[0])
    results, stats = timed_queries(conn, table, queries)
    return results, {"plan_is_seq_scan": "Seq Scan" in plan and "Index" not in plan, **stats}


def benchmark_table(conn, table: str, matrix: np.ndarray, queries: np.ndarray, ivf_lists: int | None = None, check_variability: bool = False) -> dict:
    """Carga `matrix` en `table`, mide la búsqueda exacta y barre HNSW e IVFFlat."""
    exact_ids, exact_stats = exact_baseline(conn, table, queries)
    numpy_ids = numpy_topk(matrix, queries)
    agreement = statistics.mean(recall_at_k(a, b) for a, b in zip(exact_ids, numpy_ids, strict=True))
    result = {
        "rows": len(matrix),
        "queries": len(queries),
        "exact": exact_stats | {"agreement_with_numpy": agreement},
        "hnsw": {},
        "ivfflat": {},
    }
    result["hnsw"] = build_index(conn, table, "hnsw", len(matrix)) | {"sweep": sweep(conn, table, queries, exact_ids, "hnsw", EF_SEARCH)}
    if check_variability:  # reconstruye el grafo: ¿da el mismo recall?
        first = next(p for p in result["hnsw"]["sweep"] if p["hnsw.ef_search"] == 40)["recall@10"]
        rebuilt = build_index(conn, table, "hnsw", len(matrix))
        again = sweep(conn, table, queries, exact_ids, "hnsw", (40,))[0]["recall@10"]
        result["hnsw"]["rebuild_check"] = {"recall_first_build": first, "recall_second_build": again, "build_seconds_second": rebuilt["build_seconds"]}
    ivf = build_index(conn, table, "ivfflat", len(matrix), ivf_lists)
    probes = tuple(p for p in IVF_PROBES if p <= ivf["params"]["lists"]) or (1,)
    result["ivfflat"] = ivf | {"sweep": sweep(conn, table, queries, exact_ids, "ivfflat", probes)}
    conn.execute(f"DROP INDEX IF EXISTS {table}_ann_idx")
    return result


def run_synthetic(conn, real: np.ndarray, n: int = SYNTHETIC_SIZE, n_queries: int = SYNTHETIC_QUERIES, check_variability: bool = True) -> dict:
    matrix, queries = synthetic_dataset(real, n, n_queries)
    conn.execute("CREATE TABLE synthetic (id integer PRIMARY KEY, embedding vector(384) NOT NULL)")
    start = time.perf_counter()
    copy_vectors(conn, "synthetic", matrix)
    load_seconds = time.perf_counter() - start
    conn.execute("ANALYZE synthetic")
    out = benchmark_table(conn, "synthetic", matrix, queries, check_variability=check_variability)
    out["load_seconds"] = load_seconds
    out["cosine_to_source"] = COSINE_TO_SOURCE
    return out


def run_real(conn, strategy: str) -> dict:
    """Corpus real: recall del índice y su efecto sobre el golden set (nDCG@10 y Recall@10 de artículos)."""
    from scripts.evaluate import load_chunks
    from scripts.metrics import dedupe_to_articles, evaluate_run, load_qrels, mean_metrics

    ids, matrix = embeddings.load_embeddings(f"chunks_{strategy}")
    meta = {c["chunk_id"]: (c["celex"], c["article"]) for c in load_chunks(strategy)}
    table = f"real_{strategy}"
    conn.execute(f"CREATE TABLE {table} (id integer PRIMARY KEY, celex text NOT NULL, article integer NOT NULL, embedding vector(384) NOT NULL)")
    copy_vectors(conn, table, matrix, extra=[meta[i] for i in ids])
    conn.execute(f"ANALYZE {table}")
    queries_rows = load_jsonl(QUERIES_FILE)
    embed = precomputed_query_embedder()
    queries = np.array([embed(q["query"]) for q in queries_rows], dtype=np.float32)
    lists = max(1, round(math.sqrt(len(matrix))))
    base = benchmark_table(conn, table, matrix, queries, ivf_lists=lists)
    qrels = load_qrels()

    def golden(setting: str | None, value: int | None, exact: bool) -> dict:
        (use_exact_scan if exact else use_index_scan)(conn)
        if setting:
            set_param(conn, setting, value)
        run = {}
        for q, vector in zip(queries_rows, queries, strict=True):
            sql = f"SELECT celex, article FROM {table} ORDER BY embedding <=> %s::vector LIMIT {DEPTH_END_TO_END}"
            rows = conn.execute(sql, (vector_text(vector),)).fetchall()
            run[q["query_id"]] = dedupe_to_articles([(c, a) for c, a in rows])[:10]
        means = mean_metrics(evaluate_run(run, qrels))
        return {"ndcg@10": means["ndcg@10"], "recall@10": means["recall@10"], "mrr@10": means["mrr@10"]}

    base["golden_exact"] = golden(None, None, exact=True)
    conn_kind = {"hnsw": ("hnsw.ef_search", EF_SEARCH), "ivfflat": ("ivfflat.probes", tuple(p for p in IVF_PROBES if p <= lists))}
    for kind, (setting, values) in conn_kind.items():
        build_index(conn, table, kind, len(matrix), lists)
        base[kind]["golden"] = [{setting: v, **golden(setting, v, exact=False)} for v in values]
        conn.execute(f"DROP INDEX IF EXISTS {table}_ann_idx")
    base["strategy"] = strategy
    return base


def environment(conn) -> dict:
    return {
        "postgres": conn.execute("SHOW server_version").fetchone()[0],
        "pgvector": conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()[0],
        "client_machine": platform.machine(),
        "maintenance_work_mem_session": MAINTENANCE_WORK_MEM,
        "max_parallel_maintenance_workers_session": PARALLEL_MAINTENANCE_WORKERS,
        "shared_buffers": conn.execute("SHOW shared_buffers").fetchone()[0],
    }


def run_benchmark(size: int = SYNTHETIC_SIZE, n_queries: int = SYNTHETIC_QUERIES, skip_synthetic: bool = False, output: Path = RESULTS_FILE) -> dict:
    _, real_windows = embeddings.load_embeddings("chunks_window")
    schema = f"bench_{secrets.token_hex(4)}"
    with psycopg.connect(**db.conninfo(), autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA {schema}")
        try:
            conn.execute(f"SET search_path = {schema}, public")
            report = {
                "declared_parameters": {
                    "hnsw": {"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION, "ef_search": list(EF_SEARCH)},
                    "ivfflat": {"lists": "rows/1000 (synthetic); round(sqrt(rows)) (real corpus)", "probes": list(IVF_PROBES)},
                    "ground_truth": "exact scan in Postgres on the same data (numpy cross-check)",
                    "passes": PASSES,
                    "seed": SEED,
                },
                "environment": environment(conn),
                "real": {s: run_real(conn, s) for s in ("window", "article")},
            }
            if not skip_synthetic:
                report["synthetic"] = run_synthetic(conn, real_windows, size, n_queries)
        finally:
            conn.execute("RESET search_path")
            conn.execute(f"DROP SCHEMA {schema} CASCADE")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def format_summary(report: dict) -> str:
    lines = [f"Entorno: PostgreSQL {report['environment']['postgres']}, pgvector {report['environment']['pgvector']}"]

    def table(title: str, block: dict) -> None:
        lines.append(f"\n{title}  ({block['rows']:,} filas, {block['queries']} consultas)")
        e = block["exact"]
        lines.append(f"  exacta (seq scan)           recall 1.000  mediana {e['median_ms']:7.2f} ms  p95 {e['p95_ms']:7.2f} ms  [coincidencia con numpy {e['agreement_with_numpy']:.3f}]")
        for kind, setting in (("hnsw", "hnsw.ef_search"), ("ivfflat", "ivfflat.probes")):
            b = block[kind]
            lines.append(f"  {kind}  {b['params']}  construcción {b['build_seconds']:.1f} s, índice {b['size_mb']:.1f} MB")
            for p in b["sweep"]:
                short = "" if p["queries_with_fewer_than_10_results"] == 0 else f"  [{p['queries_with_fewer_than_10_results']} consultas con <10 resultados]"
                lines.append(
                    f"    {setting.split('.')[1]:>9}={p[setting]:<4} recall@10 {p['recall@10']:.3f} [{p['recall@10_ci95'][0]:.3f}, {p['recall@10_ci95'][1]:.3f}]  "
                    f"mediana {p['median_ms']:7.2f} ms  p95 {p['p95_ms']:7.2f} ms{short}"
                )
        if "rebuild_check" in block["hnsw"]:
            r = block["hnsw"]["rebuild_check"]
            lines.append(f"  reconstrucción HNSW (ef_search=40): recall {r['recall_first_build']:.4f} -> {r['recall_second_build']:.4f}")

    for strategy, block in report["real"].items():
        table(f"CORPUS REAL · chunks '{strategy}'", block)
        g = block["golden_exact"]
        lines.append(f"  golden set — exacta: nDCG@10 {g['ndcg@10']:.3f}, Recall@10 {g['recall@10']:.3f}")
        for kind, setting in (("hnsw", "hnsw.ef_search"), ("ivfflat", "ivfflat.probes")):
            cells = "  ".join(f"{setting.split('.')[1]}={p[setting]}: {p['ndcg@10']:.3f}" for p in block[kind]["golden"])
            lines.append(f"  golden set — {kind} nDCG@10  {cells}")
    if "synthetic" in report:
        table("SINTÉTICO (datos sintéticos con la estructura de los reales)", report["synthetic"])
    return "\n".join(lines)


def run(size: int = SYNTHETIC_SIZE, n_queries: int = SYNTHETIC_QUERIES, skip_synthetic: bool = False) -> int:
    try:
        report = run_benchmark(size, n_queries, skip_synthetic)
    except (RuntimeError, FileNotFoundError, psycopg.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(format_summary(report))
    print(f"\nResultados guardados en {RESULTS_FILE.relative_to(ROOT)}")
    return 0
