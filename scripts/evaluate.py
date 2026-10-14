"""Ejecuta un sistema de recuperación sobre el golden set congelado y calcula las métricas.

Los resultados se guardan en results/<sistema>_<chunking>.json (versionado): toda cifra del README sale de aquí.
Se niega a ejecutar si el golden set no está congelado o ha sido modificado.
"""

import json
import sys
from pathlib import Path

from scripts import embeddings
from scripts.baseline_bm25 import EPSILON, K1, B, Bm25Index
from scripts.chunk_corpus import ARTICLES_FILE, WINDOWS_FILE
from scripts.dense import DenseIndex, precomputed_query_embedder
from scripts.download_corpus import ROOT, read_hashes, sha256_bytes
from scripts.fts import FtsIndex
from scripts.golden import HASH_FILE, QUERIES_FILE, check_frozen, load_jsonl
from scripts.hybrid import HybridIndex
from scripts.hybrid import K as RRF_K
from scripts.metrics import Run, dedupe_to_articles, evaluate_run, load_qrels, mean_metrics, paired_bootstrap

RESULTS_DIR = ROOT / "results"
CHUNK_FILES = {"article": ARTICLES_FILE, "window": WINDOWS_FILE}
# Cada sistema se construye con (chunking, chunks, sha256 del fichero de chunks).
SYSTEMS = {
    "bm25": lambda chunking, chunks, sha256: Bm25Index(chunks),
    "fts": lambda chunking, chunks, sha256: FtsIndex(chunking, sha256),
    "dense": lambda chunking, chunks, sha256: DenseIndex(chunking, sha256, precomputed_query_embedder()),
    "hybrid": lambda chunking, chunks, sha256: HybridIndex(chunking, sha256, precomputed_query_embedder()),
}
CONFIGS = {
    "bm25": {"k1": K1, "b": B, "epsilon": EPSILON, "tokenizer": "lowercase alphanumeric, no stemming, no stopwords"},
    "fts": {"regconfig": "english", "query": "OR of alphanumeric terms via to_tsquery", "rank": "ts_rank_cd defaults", "tie_break": "chunk_id"},
    "dense": {
        "model": embeddings.MODEL_ID,
        "revision": embeddings.REVISION,
        "query_instruction": embeddings.QUERY_INSTRUCTION,
        "similarity": "cosine (<=>) on L2-normalized vectors, exact scan, no ANN index",
        "tie_break": "chunk_id",
    },
    "hybrid": {"fusion": "RRF", "k": RRF_K, "inputs": ["fts", "dense"], "fused_at": "chunk level, then converted to distinct articles", "tie_break": "chunk_id"},
}
DEPTH = 200  # chunks recuperados antes de convertirlos a artículos
TOP_ARTICLES = 10  # artículos distintos que se evalúan
TYPES = ("paraphrase", "exact_term", "multi_article")


def load_chunks(chunking: str) -> list[dict]:
    return load_jsonl(CHUNK_FILES[chunking])


def run_system(index, queries: list[dict], depth: int = DEPTH, top: int = TOP_ARTICLES) -> Run:
    """Ranking de artículos por consulta: se recuperan `depth` chunks y se convierten a artículos distintos."""
    return {q["query_id"]: dedupe_to_articles(index.search(q["query"], depth))[:top] for q in queries}


def mean_ci(scores: dict[str, float]) -> tuple[float, float, float]:
    """Media e intervalo bootstrap del 95 % de una sola métrica (diferencia pareada frente a cero)."""
    result = paired_bootstrap(scores, dict.fromkeys(scores, 0.0))
    return result.mean_diff, result.low, result.high


def summarize(per_query: dict[str, dict[str, float]], queries: list[dict]) -> dict:
    types = {q["query_id"]: q["type"] for q in queries}
    groups = {"overall": list(per_query)} | {t: [q for q in per_query if types[q] == t] for t in TYPES}
    summary = {}
    for name, ids in groups.items():
        if not ids:  # un tipo sin consultas no se resume
            continue
        sub = {q: per_query[q] for q in ids}
        summary[name] = {"n_queries": len(ids), "mean": mean_metrics(sub), "ci95": {}}
        for metric in next(iter(sub.values())):
            _, low, high = mean_ci({q: m[metric] for q, m in sub.items()})
            summary[name]["ci95"][metric] = [low, high]
    return summary


def format_table(system: str, chunking: str, summary: dict) -> str:
    metrics = list(summary["overall"]["mean"])
    lines = [f"\n{system} · chunking={chunking}", f"{'':<14}{'n':>3}  " + "  ".join(f"{m:>9}" for m in metrics)]
    for group, data in summary.items():
        lines.append(f"{group:<14}{data['n_queries']:>3}  " + "  ".join(f"{data['mean'][m]:>9.3f}" for m in metrics))
    lines.append("\nIC 95 % (bootstrap sobre consultas)")
    for group, data in summary.items():
        cells = "  ".join(f"{m} [{data['ci95'][m][0]:.2f}, {data['ci95'][m][1]:.2f}]" for m in ("recall@10", "ndcg@10", "mrr@10"))
        lines.append(f"  {group:<14}{cells}")
    return "\n".join(lines)


def evaluate(system: str, chunking: str, results_dir: Path = RESULTS_DIR) -> dict:
    if not read_hashes(HASH_FILE):
        raise RuntimeError("el golden set no está congelado; ejecuta `python cli.py freeze-golden`")
    if problems := check_frozen():
        raise RuntimeError("; ".join(problems))
    queries = load_jsonl(QUERIES_FILE)
    chunks_sha256 = sha256_bytes(CHUNK_FILES[chunking].read_bytes())
    index = SYSTEMS[system](chunking, load_chunks(chunking), chunks_sha256)
    per_query = evaluate_run(run_system(index, queries), load_qrels())
    report = {
        "system": system,
        "chunking": chunking,
        "config": CONFIGS[system] | {"depth": DEPTH, "top_articles": TOP_ARTICLES},
        "golden_sha256": read_hashes(HASH_FILE),
        "chunks_sha256": chunks_sha256,
        "runtime": getattr(index, "info", {}),
        "summary": summarize(per_query, queries),
        "per_query": per_query,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{system}_{chunking}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run(system: str, chunking: str) -> int:
    chunkings = list(CHUNK_FILES) if chunking == "both" else [chunking]
    try:
        for c in chunkings:
            report = evaluate(system, c)
            print(format_table(system, c, report["summary"]))
    except (RuntimeError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"\nResultados guardados en {RESULTS_DIR.relative_to(ROOT)}/")
    return 0
