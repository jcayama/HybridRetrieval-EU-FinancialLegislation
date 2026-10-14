"""Genera la sección de resultados del README a partir de results/, data/gate/ y results/index_benchmark.json.

Uso:
    python cli.py report            # reescribe el bloque entre las marcas del README
    python cli.py report --check    # falla si el bloque del README no coincide con lo que generaría el script

Nada del bloque se escribe a mano. Las frases de conclusión (qué comparaciones excluyen el cero, cuáles sobreviven a
Bonferroni, qué sistema tiene la cifra más alta) se derivan de los datos, así que cambian con ellos.
"""

import json
import sys

from scripts import gate
from scripts.compare import compare
from scripts.download_corpus import ROOT
from scripts.golden import QUERIES_FILE, load_jsonl

README = ROOT / "README.md"
RESULTS_DIR = ROOT / "results"
START, END = "<!-- results:start -->", "<!-- results:end -->"

SYSTEMS = ("bm25", "fts", "dense", "hybrid", "hybrid-bm25")
CHUNKINGS = ("article", "window")
LABEL = {
    "bm25": "BM25",
    "fts": "FTS Postgres",
    "dense": "Denso (bge-small)",
    "hybrid": "Híbrido FTS + denso",
    "hybrid-bm25": "Híbrido BM25 + denso (exploratorio)",
}
CHUNK_LABEL = {"article": "artículo", "window": "ventanas"}
TYPE_LABEL = {"paraphrase": "Paráfrasis", "exact_term": "Término exacto", "multi_article": "Multi-artículo"}

# Las 15 comparaciones principales (declaradas antes de calcular) y las 6 exploratorias, como (A, B): se informa A − B.
PRIMARY = [
    ("bm25_article", "fts_article"),
    ("bm25_window", "fts_window"),
    ("fts_window", "fts_article"),
    ("bm25_window", "bm25_article"),
    ("dense_article", "bm25_article"),
    ("dense_window", "bm25_window"),
    ("dense_article", "fts_article"),
    ("dense_window", "fts_window"),
    ("dense_window", "dense_article"),
    ("hybrid_article", "dense_article"),
    ("hybrid_window", "dense_window"),
    ("hybrid_article", "fts_article"),
    ("hybrid_window", "fts_window"),
    ("hybrid_article", "bm25_article"),
    ("hybrid_window", "bm25_window"),
]
EXPLORATORY = [
    ("hybrid-bm25_article", "bm25_article"),
    ("hybrid-bm25_article", "dense_article"),
    ("hybrid-bm25_article", "hybrid_article"),
    ("hybrid-bm25_window", "bm25_window"),
    ("hybrid-bm25_window", "dense_window"),
    ("hybrid-bm25_window", "hybrid_window"),
]
FAMILY_EXPLORATORY = len(PRIMARY) + len(EXPLORATORY)  # Bonferroni sobre las 21


# ---------- formato (funciones puras) ----------


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def fmt_signed(value: float, digits: int = 3) -> str:
    return ("+" if value >= 0 else "−") + fmt(abs(value), digits)


def fmt_ci(low: float, high: float, signed: bool = False) -> str:
    f = fmt_signed if signed else fmt
    return f"[{f(low)}, {f(high)}]"


def name_of(key: str) -> str:
    system, chunking = key.rsplit("_", 1)
    return f"{LABEL[system]} ({CHUNK_LABEL[chunking]})"


def excludes_zero(low: float, high: float) -> bool:
    return low > 0 or high < 0


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def findings(rows: list[dict], family: int, best: dict[str, tuple[str, float]]) -> list[str]:
    """Frases derivadas de las filas de comparación [{'name', 'excl95', 'exclbonf'}] y de las mejores cifras."""
    out = []
    for chunking, (system, value) in best.items():
        out.append(f"Con chunking por {CHUNK_LABEL[chunking]}, la cifra global más alta de nDCG@10 entre los sistemas principales es la de **{LABEL[system]}** ({fmt(value)}).")
    out.append("Que una cifra sea la más alta **no implica** que la diferencia sea detectable: cuentan las comparaciones de abajo, no el orden de las tablas.")
    n95 = [r for r in rows if r["excl95"]]
    nb = [r for r in rows if r["exclbonf"]]
    out.append(f"De las {len(rows)} comparaciones principales, **{len(n95)}** excluyen el cero con un IC del 95 % y **{len(nb)}** lo siguen excluyendo tras la corrección de Bonferroni para {family}.")
    if nb:
        out.append("Las que sobreviven a la corrección son: " + "; ".join(f"**{r['name']}**" for r in nb) + ".")
    else:
        out.append("Ninguna comparación sobrevive a la corrección de Bonferroni.")
    rest = [r["name"] for r in n95 if not r["exclbonf"]]
    if rest:
        out.append("Excluyen el cero solo al 95 %, sin corrección (indicios, no hallazgos): " + "; ".join(rest) + ".")
    return out


def replace_block(readme: str, block: str) -> str:
    if readme.count(START) != 1 or readme.count(END) != 1 or readme.index(START) > readme.index(END):
        raise ValueError(f"el README debe contener exactamente una vez {START} y {END}, en ese orden")
    head, rest = readme.split(START)
    _, tail = rest.split(END)
    return f"{head}{START}\n{block}\n{END}{tail}"


# ---------- carga ----------


def load_results(results_dir=RESULTS_DIR) -> dict[str, dict]:
    out = {}
    for system in SYSTEMS:
        for chunking in CHUNKINGS:
            path = results_dir / f"{system}_{chunking}.json"
            if not path.exists():
                raise FileNotFoundError(f"falta {path.name}; ejecuta `python cli.py evaluate --system {system}`")
            out[f"{system}_{chunking}"] = json.loads(path.read_text(encoding="utf-8"))
    return out


# ---------- secciones ----------


def section_systems(results: dict[str, dict]) -> str:
    rows = []
    for system in SYSTEMS:
        for chunking in CHUNKINGS:
            s = results[f"{system}_{chunking}"]["summary"]["overall"]
            m, ci = s["mean"], s["ci95"]["ndcg@10"]
            rows.append([LABEL[system], CHUNK_LABEL[chunking]] + [fmt(m[k]) for k in ("recall@1", "recall@3", "recall@5", "recall@10")] + [f"{fmt(m['ndcg@10'])} {fmt_ci(*ci)}", fmt(m["mrr@10"])])
    n = results["bm25_article"]["summary"]["overall"]["n_queries"]
    header = ["Sistema", "Chunking", "Recall@1", "Recall@3", "Recall@5", "Recall@10", "nDCG@10 [IC 95 %]", "MRR@10"]
    return f"#### Métricas globales ({n} consultas)\n\n" + table(header, rows)


def section_by_type(results: dict[str, dict]) -> str:
    types = list(TYPE_LABEL)
    sample = results["bm25_article"]["summary"]
    header = ["Sistema", "Chunking"] + [f"{TYPE_LABEL[t]} ({sample[t]['n_queries']})" for t in types]
    rows = []
    for system in SYSTEMS:
        for chunking in CHUNKINGS:
            s = results[f"{system}_{chunking}"]["summary"]
            rows.append([LABEL[system], CHUNK_LABEL[chunking]] + [fmt(s[t]["mean"]["ndcg@10"]) for t in types])
    return "#### nDCG@10 por tipo de consulta\n\n" + table(header, rows)


def comparison_rows(results: dict[str, dict], pairs: list[tuple[str, str]], family: int, types: dict[str, str]) -> list[dict]:
    out = []
    for a, b in pairs:
        r = compare(results[a], results[b], "ndcg@10", types, family_size=family)
        o = r["overall"]
        bonf = next(v for k, v in r.items() if "Bonferroni" in k)
        out.append(
            {
                "name": f"{name_of(a)} − {name_of(b)}",
                "diff": o["mean_diff"],
                "ci": (o["low"], o["high"]),
                "bonf": (bonf["low"], bonf["high"]),
                "excl95": excludes_zero(o["low"], o["high"]),
                "exclbonf": excludes_zero(bonf["low"], bonf["high"]),
            }
        )
    return out


def comparison_table(rows: list[dict], family: int) -> str:
    header = ["Comparación (A − B)", "Dif. nDCG@10", "IC 95 %", f"IC Bonferroni ({family})", "Excluye 0 (95 %)", "Excluye 0 (Bonferroni)"]
    body = [[r["name"], fmt_signed(r["diff"]), fmt_ci(*r["ci"], signed=True), fmt_ci(*r["bonf"], signed=True), "sí" if r["excl95"] else "no", "**sí**" if r["exclbonf"] else "no"] for r in rows]
    return table(header, body)


def section_comparisons(results: dict[str, dict], types: dict[str, str]) -> str:
    primary = comparison_rows(results, PRIMARY, len(PRIMARY), types)
    exploratory = comparison_rows(results, EXPLORATORY, FAMILY_EXPLORATORY, types)
    best = {}
    for chunking in CHUNKINGS:
        candidates = {s: results[f"{s}_{chunking}"]["summary"]["overall"]["mean"]["ndcg@10"] for s in SYSTEMS if s != "hybrid-bm25"}
        winner = max(candidates, key=candidates.get)
        best[chunking] = (winner, candidates[winner])
    bullets = "\n".join(f"- {line}" for line in findings(primary, len(PRIMARY), best))
    exploratory_exclude = sum(r["excl95"] for r in exploratory)
    exploratory_bonf = sum(r["exclbonf"] for r in exploratory)
    return (
        f"#### Comparaciones principales (nDCG@10 global, bootstrap pareado sobre consultas)\n\n{bullets}\n\n"
        + comparison_table(primary, len(PRIMARY))
        + f"\n\n#### Comparaciones exploratorias (ideas posteriores a ver los resultados)\n\nCorrección de Bonferroni sobre {FAMILY_EXPLORATORY} (las {len(PRIMARY)} principales más estas {len(EXPLORATORY)}). "
        f"{exploratory_exclude} de {len(EXPLORATORY)} excluyen el cero al 95 % y {exploratory_bonf} tras la corrección.\n\n" + comparison_table(exploratory, FAMILY_EXPLORATORY)
    )


def section_gate() -> str:
    spec = json.loads(gate.THRESHOLDS_FILE.read_text(encoding="utf-8"))
    current = json.loads(gate.CALIBRATION_FILE.read_text(encoding="utf-8"))["current"]
    rows = [[CHUNK_LABEL[c], m, fmt(current[c][m]), fmt(spec["thresholds"][c][m])] for c in gate.CHUNKINGS for m in gate.METRICS]
    return f"#### Regression gate (margen {fmt(spec['margin'])}, {spec['margin_source']})\n\n" + table(["Chunking", "Métrica", "Valor en la calibración", "Umbral"], rows)


def section_indexes() -> str:
    bench = json.loads((RESULTS_DIR / "index_benchmark.json").read_text(encoding="utf-8"))
    blocks = [(f"Corpus real, {CHUNK_LABEL[s]}", bench["real"][s]) for s in ("window", "article")]
    if "synthetic" in bench:
        blocks.append(("Sintético", bench["synthetic"]))
    rows = []
    for title, block in blocks:
        e = block["exact"]
        thousands = f"{block['rows']:,}".replace(",", ".")  # solo el número: el título puede llevar comas
        rows.append([f"{title} ({thousands} filas)", "Exacta (referencia)", "—", "1,000", fmt(e["median_ms"], 2), "—", "—"])
        for kind, setting, wanted in (("hnsw", "hnsw.ef_search", (40,)), ("ivfflat", "ivfflat.probes", (1, 10))):
            params = block[kind]["params"]
            for p in block[kind]["sweep"]:
                if p[setting] in wanted:
                    short = "ef_search" if kind == "hnsw" else "probes"
                    label = f"{'HNSW' if kind == 'hnsw' else 'IVFFlat'} ({short}={p[setting]})"
                    shown = ", ".join(f"{k}={v}" for k, v in params.items())
                    rows.append(["", label, shown, fmt(p["recall@10"]), fmt(p["median_ms"], 2), fmt(block[kind]["build_seconds"], 1), fmt(block[kind]["size_mb"], 1)])
    return "#### Índices aproximados (recall@10 frente a la búsqueda exacta y latencia mediana)\n\n" + table(["Conjunto", "Método", "Parámetros", "Recall@10", "Latencia (ms)", "Construcción (s)", "Índice (MB)"], rows)


def render(results: dict[str, dict], types: dict[str, str]) -> str:
    return "\n\n".join([section_systems(results), section_by_type(results), section_comparisons(results, types), section_gate(), section_indexes()])


def run(check: bool = False) -> int:
    try:
        types = {q["query_id"]: q["type"] for q in load_jsonl(QUERIES_FILE)}
        block = render(load_results(), types)
        readme = README.read_text(encoding="utf-8")
        updated = replace_block(readme, block)
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if check:
        if updated != readme:
            print("ERROR: la sección de resultados del README está desactualizada; ejecuta `python cli.py report`", file=sys.stderr)
            return 1
        print("La sección de resultados del README está al día.")
        return 0
    README.write_text(updated, encoding="utf-8")
    print(f"Sección de resultados escrita en {README.relative_to(ROOT)} ({len(block.splitlines())} líneas).")
    return 0
