"""Generates the generated blocks of the READMEs (English `README.md`, Spanish `README.es.md`).

Genera los bloques generados de los README (inglés `README.md`, español `README.es.md`) a partir de results/,
data/gate/ y results/index_benchmark.json.

Uso:
    python cli.py report            # reescribe los bloques entre las marcas de ambos README
    python cli.py report --check    # falla si algún bloque no coincide con lo que generaría el script

Dos bloques por README: `results` (las tablas completas) y `tldr` (las cifras clave del resumen inicial). Nada de
ellos se escribe a mano. Las frases de conclusión (qué comparaciones excluyen el cero, cuáles sobreviven a
Bonferroni, qué sistema tiene la cifra más alta) se derivan de los datos, así que cambian con ellos.
"""

import json
import sys

from scripts import gate
from scripts.compare import compare
from scripts.download_corpus import ROOT
from scripts.golden import QUERIES_FILE, load_jsonl

LANGS = ("en", "es")
DEFAULT_LANG = "en"
README_FILES = {"en": ROOT / "README.md", "es": ROOT / "README.es.md"}
RESULTS_DIR = ROOT / "results"
START, END = "<!-- results:start -->", "<!-- results:end -->"
TLDR_START, TLDR_END = "<!-- tldr:start -->", "<!-- tldr:end -->"

SYSTEMS = ("bm25", "fts", "dense", "hybrid", "hybrid-bm25")
MAIN_SYSTEMS = ("bm25", "fts", "dense", "hybrid")  # los principales; hybrid-bm25 es exploratorio
CHUNKINGS = ("article", "window")
TYPES = ("paraphrase", "exact_term", "multi_article")

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

# ---------- paquetes de idioma ----------

PACKS = {
    "es": {
        "decimal": ",",
        "thousands": ".",
        "yes": "sí",
        "no": "no",
        "system": {
            "bm25": "BM25",
            "fts": "FTS Postgres",
            "dense": "Denso (bge-small)",
            "hybrid": "Híbrido FTS + denso",
            "hybrid-bm25": "Híbrido BM25 + denso (exploratorio)",
        },
        "chunk": {"article": "artículo", "window": "ventanas"},
        "type": {"paraphrase": "Paráfrasis", "exact_term": "Término exacto", "multi_article": "Multi-artículo"},
        "manual_margin": {"decisión manual": "decisión manual", "regla declarada (3 × ruido)": "regla declarada (3 × ruido)"},
        "systems_title": "#### Métricas globales ({n} consultas)",
        "systems_header": ["Sistema", "Chunking", "Recall@1", "Recall@3", "Recall@5", "Recall@10", "nDCG@10 [IC 95 %]", "MRR@10"],
        "by_type_title": "#### nDCG@10 por tipo de consulta",
        "by_type_header": ["Sistema", "Chunking"],
        "primary_title": "#### Comparaciones principales (nDCG@10 global, bootstrap pareado sobre consultas)",
        "exploratory_title": "#### Comparaciones exploratorias (ideas posteriores a ver los resultados)",
        "exploratory_intro": "Corrección de Bonferroni sobre {family} (las {np} principales más estas {ne}). {a} de {ne} excluyen el cero al 95 % y {b} tras la corrección.",
        "comparison_header": ["Comparación (A − B)", "Dif. nDCG@10", "IC 95 %", "IC Bonferroni ({family})", "Excluye 0 (95 %)", "Excluye 0 (Bonferroni)"],
        "best": "Con chunking por {chunk}, la cifra global más alta de nDCG@10 entre los sistemas principales es la de **{system}** ({value}).",
        "not_imply": "Que una cifra sea la más alta **no implica** que la diferencia sea detectable: cuentan las comparaciones de abajo, no el orden de las tablas.",
        "counts": "De las {n} comparaciones principales, **{a}** excluyen el cero con un IC del 95 % y **{b}** lo siguen excluyendo tras la corrección de Bonferroni para {family}.",
        "survivors": "Las que sobreviven a la corrección son: {names}.",
        "no_survivors": "Ninguna comparación sobrevive a la corrección de Bonferroni.",
        "hints": "Excluyen el cero solo al 95 %, sin corrección (indicios, no hallazgos): {names}.",
        "gate_title": "#### Regression gate (margen {margin}, {source})",
        "gate_header": ["Chunking", "Métrica", "Valor en la calibración", "Umbral"],
        "index_title": "#### Índices aproximados (recall@10 frente a la búsqueda exacta y latencia mediana)",
        "index_header": ["Conjunto", "Método", "Parámetros", "Recall@10", "Latencia (ms)", "Construcción (s)", "Índice (MB)"],
        "real_corpus": "Corpus real, {chunk}",
        "synthetic": "Sintético",
        "rows_suffix": "{thousands} filas",
        "exact_reference": "Exacta (referencia)",
        "tldr_best": "**Cifra global más alta de nDCG@10 entre los sistemas principales:** {system}, con {window} con ventanas ({article} con artículos), sobre {n} consultas.",
        "tldr_survivors": "**Las únicas comparaciones que sobreviven a la corrección por comparaciones múltiples:** {names}. Cualquier otra diferencia entre los sistemas principales no es detectable con {n} consultas.",
        "tldr_none": "**Ninguna diferencia entre los sistemas principales sobrevive a la corrección por comparaciones múltiples** con {n} consultas.",
        "tldr_types_mixed": "**Ningún sistema gana en todos los tipos de consulta** (ventanas, nDCG@10): {pairs}.",
        "tldr_types_same": "**Un mismo sistema gana en todos los tipos de consulta** (ventanas, nDCG@10): {system}.",
        "tldr_table_title": "nDCG@10 y Recall@10 con ventanas:",
        "tldr_table_header": ["Sistema", "nDCG@10", "Recall@10"],
    },
    "en": {
        "decimal": ".",
        "thousands": ",",
        "yes": "yes",
        "no": "no",
        "system": {
            "bm25": "BM25",
            "fts": "Postgres FTS",
            "dense": "Dense (bge-small)",
            "hybrid": "Hybrid FTS + dense",
            "hybrid-bm25": "Hybrid BM25 + dense (exploratory)",
        },
        "chunk": {"article": "article", "window": "window"},
        "type": {"paraphrase": "Paraphrase", "exact_term": "Exact term", "multi_article": "Multi-article"},
        "manual_margin": {"decisión manual": "manual decision", "regla declarada (3 × ruido)": "declared rule (3 × noise)"},
        "systems_title": "#### Overall metrics ({n} queries)",
        "systems_header": ["System", "Chunking", "Recall@1", "Recall@3", "Recall@5", "Recall@10", "nDCG@10 [95% CI]", "MRR@10"],
        "by_type_title": "#### nDCG@10 by query type",
        "by_type_header": ["System", "Chunking"],
        "primary_title": "#### Primary comparisons (overall nDCG@10, paired bootstrap over queries)",
        "exploratory_title": "#### Exploratory comparisons (ideas that came after seeing the results)",
        "exploratory_intro": "Bonferroni correction over {family} (the {np} primary comparisons plus these {ne}). {a} of {ne} exclude zero at 95% and {b} after the correction.",
        "comparison_header": ["Comparison (A − B)", "nDCG@10 diff.", "95% CI", "Bonferroni CI ({family})", "Excludes 0 (95%)", "Excludes 0 (Bonferroni)"],
        "best": "With {chunk} chunking, the highest overall nDCG@10 among the main systems is **{system}** ({value}).",
        "not_imply": "Having the highest figure does **not imply** that the difference is detectable: what counts are the comparisons below, not the order of the tables.",
        "counts": "Of the {n} primary comparisons, **{a}** exclude zero with a 95% CI and **{b}** still exclude it after the Bonferroni correction for {family}.",
        "survivors": "Those that survive the correction are: {names}.",
        "no_survivors": "No comparison survives the Bonferroni correction.",
        "hints": "Exclude zero only at 95%, without correction (hints, not findings): {names}.",
        "gate_title": "#### Regression gate (margin {margin}, {source})",
        "gate_header": ["Chunking", "Metric", "Value at calibration", "Threshold"],
        "index_title": "#### Approximate indexes (recall@10 against exact search, and median latency)",
        "index_header": ["Dataset", "Method", "Parameters", "Recall@10", "Latency (ms)", "Build (s)", "Index (MB)"],
        "real_corpus": "Real corpus, {chunk}",
        "synthetic": "Synthetic",
        "rows_suffix": "{thousands} rows",
        "exact_reference": "Exact (reference)",
        "tldr_best": "**Highest overall nDCG@10 among the main systems:** {system}, with {window} using windows ({article} using articles), over {n} queries.",
        "tldr_survivors": "**The only comparisons that survive the multiple-comparison correction:** {names}. Every other difference between the main systems is not detectable with {n} queries.",
        "tldr_none": "**No difference between the main systems survives the multiple-comparison correction** with {n} queries.",
        "tldr_types_mixed": "**No single system wins on every query type** (window chunking, nDCG@10): {pairs}.",
        "tldr_types_same": "**One system wins on every query type** (window chunking, nDCG@10): {system}.",
        "tldr_table_title": "nDCG@10 and Recall@10 with window chunking:",
        "tldr_table_header": ["System", "nDCG@10", "Recall@10"],
    },
}


# ---------- formato (funciones puras) ----------


def fmt(value: float, digits: int = 3, lang: str = DEFAULT_LANG) -> str:
    return f"{value:.{digits}f}".replace(".", PACKS[lang]["decimal"])


def fmt_signed(value: float, digits: int = 3, lang: str = DEFAULT_LANG) -> str:
    return ("+" if value >= 0 else "−") + fmt(abs(value), digits, lang)


def fmt_ci(low: float, high: float, signed: bool = False, lang: str = DEFAULT_LANG) -> str:
    f = fmt_signed if signed else fmt
    return f"[{f(low, 3, lang)}, {f(high, 3, lang)}]"


def name_of(key: str, lang: str = DEFAULT_LANG) -> str:
    system, chunking = key.rsplit("_", 1)
    return f"{PACKS[lang]['system'][system]} ({PACKS[lang]['chunk'][chunking]})"


def excludes_zero(low: float, high: float) -> bool:
    return low > 0 or high < 0


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def findings(rows: list[dict], family: int, best: dict[str, tuple[str, float]], lang: str = DEFAULT_LANG) -> list[str]:
    """Frases derivadas de las filas de comparación [{'name', 'excl95', 'exclbonf'}] y de las mejores cifras."""
    t = PACKS[lang]
    out = [t["best"].format(chunk=t["chunk"][chunking], system=t["system"][system], value=fmt(value, 3, lang)) for chunking, (system, value) in best.items()]
    out.append(t["not_imply"])
    n95 = [r for r in rows if r["excl95"]]
    nb = [r for r in rows if r["exclbonf"]]
    out.append(t["counts"].format(n=len(rows), a=len(n95), b=len(nb), family=family))
    out.append(t["survivors"].format(names="; ".join(f"**{r['name']}**" for r in nb)) if nb else t["no_survivors"])
    rest = [r["name"] for r in n95 if not r["exclbonf"]]
    if rest:
        out.append(t["hints"].format(names="; ".join(rest)))
    return out


def replace_block(readme: str, block: str, start: str = START, end: str = END) -> str:
    if readme.count(start) != 1 or readme.count(end) != 1 or readme.index(start) > readme.index(end):
        raise ValueError(f"el README debe contener exactamente una vez {start} y {end}, en ese orden")
    head, rest = readme.split(start)
    _, tail = rest.split(end)
    return f"{head}{start}\n{block}\n{end}{tail}"


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


# ---------- secciones de «results» ----------


def section_systems(results: dict[str, dict], lang: str = DEFAULT_LANG) -> str:
    t = PACKS[lang]
    rows = []
    for system in SYSTEMS:
        for chunking in CHUNKINGS:
            s = results[f"{system}_{chunking}"]["summary"]["overall"]
            m, ci = s["mean"], s["ci95"]["ndcg@10"]
            rows.append(
                [t["system"][system], t["chunk"][chunking]]
                + [fmt(m[k], 3, lang) for k in ("recall@1", "recall@3", "recall@5", "recall@10")]
                + [f"{fmt(m['ndcg@10'], 3, lang)} {fmt_ci(*ci, lang=lang)}", fmt(m["mrr@10"], 3, lang)]
            )
    n = results["bm25_article"]["summary"]["overall"]["n_queries"]
    return t["systems_title"].format(n=n) + "\n\n" + table(t["systems_header"], rows)


def section_by_type(results: dict[str, dict], lang: str = DEFAULT_LANG) -> str:
    t = PACKS[lang]
    sample = results["bm25_article"]["summary"]
    header = t["by_type_header"] + [f"{t['type'][k]} ({sample[k]['n_queries']})" for k in TYPES]
    rows = []
    for system in SYSTEMS:
        for chunking in CHUNKINGS:
            s = results[f"{system}_{chunking}"]["summary"]
            rows.append([t["system"][system], t["chunk"][chunking]] + [fmt(s[k]["mean"]["ndcg@10"], 3, lang) for k in TYPES])
    return t["by_type_title"] + "\n\n" + table(header, rows)


def comparison_rows(results: dict[str, dict], pairs: list[tuple[str, str]], family: int, types: dict[str, str], lang: str = DEFAULT_LANG) -> list[dict]:
    out = []
    for a, b in pairs:
        r = compare(results[a], results[b], "ndcg@10", types, family_size=family)
        o = r["overall"]
        bonf = next(v for k, v in r.items() if "Bonferroni" in k)
        out.append(
            {
                "name": f"{name_of(a, lang)} − {name_of(b, lang)}",
                "diff": o["mean_diff"],
                "ci": (o["low"], o["high"]),
                "bonf": (bonf["low"], bonf["high"]),
                "excl95": excludes_zero(o["low"], o["high"]),
                "exclbonf": excludes_zero(bonf["low"], bonf["high"]),
            }
        )
    return out


def comparison_table(rows: list[dict], family: int, lang: str = DEFAULT_LANG) -> str:
    t = PACKS[lang]
    header = [h.format(family=family) for h in t["comparison_header"]]
    yes, no = t["yes"], t["no"]
    body = [
        [r["name"], fmt_signed(r["diff"], 3, lang), fmt_ci(*r["ci"], signed=True, lang=lang), fmt_ci(*r["bonf"], signed=True, lang=lang), yes if r["excl95"] else no, f"**{yes}**" if r["exclbonf"] else no]
        for r in rows
    ]
    return table(header, body)


def best_by_chunking(results: dict[str, dict]) -> dict[str, tuple[str, float]]:
    best = {}
    for chunking in CHUNKINGS:
        candidates = {s: results[f"{s}_{chunking}"]["summary"]["overall"]["mean"]["ndcg@10"] for s in MAIN_SYSTEMS}
        winner = max(candidates, key=candidates.get)
        best[chunking] = (winner, candidates[winner])
    return best


def section_comparisons(results: dict[str, dict], types: dict[str, str], lang: str = DEFAULT_LANG) -> str:
    t = PACKS[lang]
    primary = comparison_rows(results, PRIMARY, len(PRIMARY), types, lang)
    exploratory = comparison_rows(results, EXPLORATORY, FAMILY_EXPLORATORY, types, lang)
    bullets = "\n".join(f"- {line}" for line in findings(primary, len(PRIMARY), best_by_chunking(results), lang))
    intro = t["exploratory_intro"].format(
        family=FAMILY_EXPLORATORY, np=len(PRIMARY), ne=len(EXPLORATORY), a=sum(r["excl95"] for r in exploratory), b=sum(r["exclbonf"] for r in exploratory)
    )
    return (
        f"{t['primary_title']}\n\n{bullets}\n\n"
        + comparison_table(primary, len(PRIMARY), lang)
        + f"\n\n{t['exploratory_title']}\n\n{intro}\n\n"
        + comparison_table(exploratory, FAMILY_EXPLORATORY, lang)
    )


def section_gate(lang: str = DEFAULT_LANG) -> str:
    t = PACKS[lang]
    spec = json.loads(gate.THRESHOLDS_FILE.read_text(encoding="utf-8"))
    current = json.loads(gate.CALIBRATION_FILE.read_text(encoding="utf-8"))["current"]
    rows = [[t["chunk"][c], m, fmt(current[c][m], 3, lang), fmt(spec["thresholds"][c][m], 3, lang)] for c in gate.CHUNKINGS for m in gate.METRICS]
    source = t["manual_margin"].get(spec["margin_source"], spec["margin_source"])
    return t["gate_title"].format(margin=fmt(spec["margin"], 3, lang), source=source) + "\n\n" + table(t["gate_header"], rows)


def section_indexes(lang: str = DEFAULT_LANG) -> str:
    t = PACKS[lang]
    bench = json.loads((RESULTS_DIR / "index_benchmark.json").read_text(encoding="utf-8"))
    blocks = [(t["real_corpus"].format(chunk=t["chunk"][s]), bench["real"][s]) for s in ("window", "article")]
    if "synthetic" in bench:
        blocks.append((t["synthetic"], bench["synthetic"]))
    rows = []
    for title, block in blocks:
        e = block["exact"]
        thousands = f"{block['rows']:,}".replace(",", t["thousands"])  # solo el número: el título puede llevar comas
        rows.append([f"{title} ({t['rows_suffix'].format(thousands=thousands)})", t["exact_reference"], "—", fmt(1.0, 3, lang), fmt(e["median_ms"], 2, lang), "—", "—"])
        for kind, setting, wanted in (("hnsw", "hnsw.ef_search", (40,)), ("ivfflat", "ivfflat.probes", (1, 10))):
            params = block[kind]["params"]
            for p in block[kind]["sweep"]:
                if p[setting] in wanted:
                    short = "ef_search" if kind == "hnsw" else "probes"
                    label = f"{'HNSW' if kind == 'hnsw' else 'IVFFlat'} ({short}={p[setting]})"
                    shown = ", ".join(f"{k}={v}" for k, v in params.items())
                    rows.append(["", label, shown, fmt(p["recall@10"], 3, lang), fmt(p["median_ms"], 2, lang), fmt(block[kind]["build_seconds"], 1, lang), fmt(block[kind]["size_mb"], 1, lang)])
    return t["index_title"] + "\n\n" + table(t["index_header"], rows)


def render(results: dict[str, dict], types: dict[str, str], lang: str = DEFAULT_LANG) -> str:
    return "\n\n".join([section_systems(results, lang), section_by_type(results, lang), section_comparisons(results, types, lang), section_gate(lang), section_indexes(lang)])


# ---------- bloque «tldr» (cifras clave del resumen inicial) ----------


def render_headline(results: dict[str, dict], types: dict[str, str], lang: str = DEFAULT_LANG) -> str:
    t = PACKS[lang]
    n = results["bm25_article"]["summary"]["overall"]["n_queries"]
    best = best_by_chunking(results)
    window_system, window_value = best["window"]
    article_value = results[f"{window_system}_article"]["summary"]["overall"]["mean"]["ndcg@10"]
    bullets = [t["tldr_best"].format(system=t["system"][window_system], window=fmt(window_value, 3, lang), article=fmt(article_value, 3, lang), n=n)]

    primary = comparison_rows(results, PRIMARY, len(PRIMARY), types, lang)
    survivors = [r for r in primary if r["exclbonf"]]
    if survivors:
        names = "; ".join(f"{r['name']} ({fmt_signed(r['diff'], 3, lang)} nDCG@10)" for r in survivors)
        bullets.append(t["tldr_survivors"].format(names=names, n=n))
    else:
        bullets.append(t["tldr_none"].format(n=n))

    winners = {}
    for kind in TYPES:
        scores = {s: results[f"{s}_window"]["summary"][kind]["mean"]["ndcg@10"] for s in MAIN_SYSTEMS}
        winner = max(scores, key=scores.get)
        winners[kind] = (winner, scores[winner])
    if len({w for w, _ in winners.values()}) > 1:
        pairs = "; ".join(f"{t['type'][k].lower()} → {t['system'][w]} ({fmt(v, 3, lang)})" for k, (w, v) in winners.items())
        bullets.append(t["tldr_types_mixed"].format(pairs=pairs))
    else:
        bullets.append(t["tldr_types_same"].format(system=t["system"][next(iter(winners.values()))[0]]))

    rows = [[t["system"][s], fmt(results[f"{s}_window"]["summary"]["overall"]["mean"]["ndcg@10"], 3, lang), fmt(results[f"{s}_window"]["summary"]["overall"]["mean"]["recall@10"], 3, lang)] for s in SYSTEMS]
    return "\n".join(f"- {b}" for b in bullets) + f"\n\n{t['tldr_table_title']}\n\n" + table(t["tldr_table_header"], rows)


# ---------- ejecución ----------


def run(check: bool = False) -> int:
    try:
        types = {q["query_id"]: q["type"] for q in load_jsonl(QUERIES_FILE)}
        results = load_results()
        outputs: dict[str, tuple[str, str]] = {}
        for lang in LANGS:
            readme = README_FILES[lang].read_text(encoding="utf-8")
            updated = replace_block(readme, render(results, types, lang), START, END)
            updated = replace_block(updated, render_headline(results, types, lang), TLDR_START, TLDR_END)
            outputs[lang] = (readme, updated)
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    stale = [README_FILES[lang].name for lang, (old, new) in outputs.items() if old != new]
    if check:
        if stale:
            print(f"ERROR: los bloques generados de {', '.join(stale)} están desactualizados; ejecuta `python cli.py report`", file=sys.stderr)
            return 1
        print(f"Los bloques generados de {' y '.join(README_FILES[lang].name for lang in LANGS)} están al día.")
        return 0
    for lang, (_, new) in outputs.items():
        README_FILES[lang].write_text(new, encoding="utf-8")
    print(f"Bloques generados escritos en {' y '.join(README_FILES[lang].name for lang in LANGS)}.")
    return 0
