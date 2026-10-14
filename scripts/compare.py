"""Comparación pareada de dos resultados guardados en results/ (diferencia A − B con intervalo bootstrap).

Uso: python cli.py compare bm25_window fts_window [--metric ndcg@10]

Solo compara resultados obtenidos sobre el mismo golden set congelado. Las comparaciones principales deben
declararse antes de calcular; el resto son descriptivas (con muchos intervalos, alguno excluirá el cero por azar).
"""

import json
import sys
from pathlib import Path

from scripts.download_corpus import ROOT
from scripts.evaluate import RESULTS_DIR, TYPES
from scripts.golden import QUERIES_FILE, load_jsonl
from scripts.metrics import paired_bootstrap


def load_result(name: str, results_dir: Path = RESULTS_DIR) -> dict:
    path = results_dir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no existe {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}; ejecuta `python cli.py evaluate`")
    return json.loads(path.read_text(encoding="utf-8"))


def compare(report_a: dict, report_b: dict, metric: str, query_types: dict[str, str]) -> dict[str, dict]:
    if report_a["golden_sha256"] != report_b["golden_sha256"]:
        raise ValueError("los resultados se calcularon sobre golden sets distintos")
    if set(report_a["per_query"]) != set(report_b["per_query"]):
        raise ValueError("los resultados no cubren las mismas consultas")
    groups = {"overall": list(report_a["per_query"])} | {t: [q for q in report_a["per_query"] if query_types[q] == t] for t in TYPES}
    out = {}
    for name, ids in groups.items():
        if not ids:
            continue
        a = {q: report_a["per_query"][q][metric] for q in ids}
        b = {q: report_b["per_query"][q][metric] for q in ids}
        boot = paired_bootstrap(a, b)
        out[name] = {
            "n": len(ids),
            "mean_diff": boot.mean_diff,
            "low": boot.low,
            "high": boot.high,
            "a_better": sum(a[q] > b[q] for q in ids),
            "b_better": sum(a[q] < b[q] for q in ids),
            "ties": sum(a[q] == b[q] for q in ids),
        }
    return out


def format_comparison(name_a: str, name_b: str, metric: str, result: dict) -> str:
    lines = [f"\n{metric}: {name_a} − {name_b}  (IC 95 % bootstrap pareado sobre consultas)"]
    for group, r in result.items():
        lines.append(
            f"  {group:<14} n={r['n']:>2}  dif={r['mean_diff']:+.3f}  IC [{r['low']:+.3f}, {r['high']:+.3f}]  "
            f"{name_a} mejor en {r['a_better']}, {name_b} mejor en {r['b_better']}, igual en {r['ties']}"
        )
    return "\n".join(lines)


def run(name_a: str, name_b: str, metric: str) -> int:
    try:
        a, b = load_result(name_a), load_result(name_b)
        if metric not in next(iter(a["per_query"].values())):
            raise ValueError(f"métrica desconocida: {metric}")
        types = {q["query_id"]: q["type"] for q in load_jsonl(QUERIES_FILE)}
        print(format_comparison(name_a, name_b, metric, compare(a, b, metric, types)))
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0
