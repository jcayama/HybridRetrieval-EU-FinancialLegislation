"""Análisis de errores del sistema principal (híbrido FTS + denso): por qué falla cada consulta mala.

Decisiones fijadas antes de ejecutar el análisis:
- Consulta mala = nDCG@10 < BAD_NDCG o ningún artículo relevante (grado >= 2) en el top-10, con el híbrido y para cada
  chunking por separado. La definición se aplica a `results/`, no se eligen casos a mano.
- Por cada caso: posición del artículo relevante en las listas de FTS y denso (a nivel de chunk) y en el híbrido, lemas
  de la consulta presentes o ausentes en el artículo, y en cuántos chunks aparece cada lema.
- Sondas de contraste (PROBES): cada una cambia UNA sola cosa respecto a la consulta original y se redactó sabiendo cuál
  es el artículo relevante. Sirven para mostrar que una causa es SUFICIENTE para explicar el fallo; no prueban que ese
  cambio mejore el sistema, y no se usan para tocar el sistema ni el golden set.
- Autoverificación: el nDCG@10 que se recalcula aquí tiene que coincidir con el guardado en `results/`.
"""

import json
import sys
from collections import defaultdict

from scripts import db, embeddings, rrf
from scripts.chunk_corpus import ARTICLES_FILE
from scripts.dense import DenseIndex, precomputed_query_embedder
from scripts.download_corpus import ROOT, sha256_bytes
from scripts.evaluate import CHUNK_FILES, DEPTH, RESULTS_DIR, TOP_ARTICLES
from scripts.fts import PUBLIC_ONLY, SEARCH_SQL, FtsIndex
from scripts.golden import QUERIES_FILE, check_frozen, load_jsonl
from scripts.hybrid import K as RRF_K
from scripts.metrics import Article, Judgments, dedupe_to_articles, load_qrels, ndcg_at_k, recall_at_k

RESULTS_FILE = RESULTS_DIR / "error_analysis.json"
SOURCE_SYSTEM = "hybrid"
CHUNKINGS = ("article", "window")
BAD_NDCG = 0.3
EFFECT_STRONG = 0.1  # mejora del nDCG@10 del híbrido a partir de la cual una sonda «explica» el fallo
TOLERANCE = 1e-9
ACTS = {"32011L0061": "AIFMD", "32022R2554": "DORA", "32024R1689": "AI Act"}

# kind = "query": se sustituye el texto de la consulta (la usan FTS y denso); kind = "fts_tsquery": solo cambia la
# tsquery de FTS (el denso sigue con la consulta original).
PROBES = {
    "q006": {"kind": "query", "value": "How are the ICT third-party service providers that financial firms depend on most identified and supervised?"},
    "q018": {"kind": "fts_tsquery", "value": "stress & test"},
    "q028": {"kind": "query", "value": "How large can the fines be for companies that break the rules?"},
    "q029": {"kind": "query", "value": "Can companies try out new AI systems in a regulatory sandbox under official supervision before placing them on the market?"},
    "q036": {"kind": "fts_tsquery", "value": "regist <2> inform"},
}


# ---------- funciones puras ----------


def is_bad(metrics: dict[str, float]) -> bool:
    """Mala = nDCG@10 por debajo del umbral o ningún relevante (grado >= 2) en el top-10."""
    return metrics["ndcg@10"] < BAD_NDCG or metrics["recall@10"] == 0


def select_bad(per_query: dict[str, dict[str, float]]) -> list[str]:
    return sorted(query_id for query_id, metrics in per_query.items() if is_bad(metrics))


def position(ranking: list, target) -> int | None:
    """Posición (desde 1) de `target` en `ranking`, o None si no está."""
    for index, item in enumerate(ranking, 1):
        if item == target:
            return index
    return None


def best_chunk_position(chunks: list[tuple[str, Article]], article: Article) -> int | None:
    """Posición del mejor chunk de `article` en una lista de (chunk_id, artículo)."""
    return position([a for _, a in chunks], article)


def lemma_split(query_lemmas: list[str], article_lemmas: set[str]) -> tuple[list[str], list[str]]:
    """(presentes, ausentes) de los lemas de la consulta en el artículo, conservando el orden."""
    return [x for x in query_lemmas if x in article_lemmas], [x for x in query_lemmas if x not in article_lemmas]


def probe_effect(ndcg_before: float, ndcg_after: float) -> str:
    """Etiqueta fija: «explains» si el nDCG@10 mejora al menos EFFECT_STRONG; «partial» si mejora menos; «none» si no mejora."""
    gain = ndcg_after - ndcg_before
    return "explains" if gain >= EFFECT_STRONG else "partial" if gain > TOLERANCE else "none"


def label(article: Article) -> str:
    return f"{ACTS[article[0]]} {article[1]}"


def hybrid_ranking(lexical: list[tuple[str, Article]], semantic: list[tuple[str, Article]]) -> list[Article]:
    """Mismo procedimiento que `RrfIndex.search`: RRF por chunk_id, DEPTH chunks y después artículos distintos."""
    articles = dict(lexical) | dict(semantic)
    fused = rrf.rrf([[c for c, _ in lexical], [c for c, _ in semantic]], k=RRF_K)
    return dedupe_to_articles([articles[c] for c in fused[:DEPTH]])


# ---------- con Postgres ----------


class Analyzer:
    """Rankings de FTS, denso e híbrido para un chunking, con los mismos índices que `evaluate`."""

    def __init__(self, chunking: str):
        self.chunking = chunking
        sha = sha256_bytes(CHUNK_FILES[chunking].read_bytes())
        self.fts = FtsIndex(chunking, sha)
        self._precomputed = precomputed_query_embedder()
        self._model = None
        self.dense = DenseIndex(chunking, sha, self._embed)
        self.articles = {(r["celex"], r["article"]): r["text"] for r in load_jsonl(ARTICLES_FILE)}

    def _embed(self, text: str):
        """Vectores precalculados para las consultas del golden set (como en `evaluate`); el modelo, solo para las sondas."""
        try:
            return self._precomputed(text)
        except KeyError:
            if self._model is None:
                try:
                    self._model = embeddings.load_model()
                except ImportError as error:
                    raise RuntimeError("las sondas con consultas nuevas necesitan `pip install -r requirements-embeddings.txt`") from error
            return embeddings.encode(self._model, embeddings.texts_for("queries", [{"query": text}]))[0]

    def fts_chunks(self, query: str, tsquery: str | None = None) -> list[tuple[str, Article]]:
        if tsquery is None:
            return self.fts.search_chunks(query, DEPTH)
        with db.connect() as conn:
            rows = conn.execute(SEARCH_SQL.format(scope=PUBLIC_ONLY), {"query": tsquery, "strategy": self.chunking, "depth": DEPTH}).fetchall()
        return [(chunk_id, (celex, article)) for chunk_id, celex, article in rows]

    def rankings(self, query: str, tsquery: str | None = None) -> dict:
        lexical = self.fts_chunks(query, tsquery)
        semantic = self.dense.search_chunks(query, DEPTH)
        return {"fts": lexical, "dense": semantic, "hybrid": hybrid_ranking(lexical, semantic)}

    def lemmas(self, text: str) -> list[str]:
        with db.connect() as conn:
            return [r[0] for r in conn.execute("SELECT lexeme FROM unnest(to_tsvector('english', %s))", (text,)).fetchall()]

    def chunk_frequency(self, lemma: str) -> int:
        """Nº de chunks de este chunking que contienen el lema. El lema ya está procesado: no se vuelve a lematizar."""
        with db.connect() as conn:
            sql = "SELECT count(*) FROM chunks WHERE strategy = %s AND tenant_id IS NULL AND tsv @@ (quote_literal(%s))::tsquery"
            return conn.execute(sql, (self.chunking, lemma)).fetchone()[0]


def positions(ranks: dict, judgments: Judgments) -> dict[str, dict]:
    return {
        label(article): {
            "grade": grade,
            "fts": best_chunk_position(ranks["fts"], article),
            "dense": best_chunk_position(ranks["dense"], article),
            "hybrid": position(ranks["hybrid"], article),
        }
        for article, grade in sorted(judgments.items())
    }


def analyze_case(analyzer: Analyzer, query: dict, judgments: Judgments, saved: dict[str, float]) -> dict:
    """Un caso (consulta mala, chunking): evidencia sobre el original y resultado de su sonda."""
    ranks = analyzer.rankings(query["query"])
    top = ranks["hybrid"][:TOP_ARTICLES]
    ndcg = ndcg_at_k(top, judgments, 10)
    if abs(ndcg - saved["ndcg@10"]) > TOLERANCE or abs(recall_at_k(top, judgments, 10) - saved["recall@10"]) > TOLERANCE:
        raise RuntimeError(f"{query['query_id']} ({analyzer.chunking}): el ranking regenerado no reproduce las métricas guardadas en results/")
    query_lemmas = analyzer.lemmas(query["query"])
    relevant = [a for a, g in judgments.items() if g >= 2] or list(judgments)
    lexical_winner = dedupe_to_articles([a for _, a in ranks["fts"]])[0]
    case = {
        "query_id": query["query_id"],
        "type": query["type"],
        "query": query["query"],
        "chunking": analyzer.chunking,
        "ndcg@10": ndcg,
        "recall@10": saved["recall@10"],
        "positions": positions(ranks, judgments),
        "hybrid_top5": [label(a) for a in ranks["hybrid"][:5]],
        "lemma_chunk_frequency": {x: analyzer.chunk_frequency(x) for x in query_lemmas},
        "lemmas_in_relevant": {label(a): dict(zip(("present", "absent"), lemma_split(query_lemmas, set(analyzer.lemmas(analyzer.articles[a]))), strict=True)) for a in relevant[:2]},
        "lexical_winner": {"article": label(lexical_winner), "present": lemma_split(query_lemmas, set(analyzer.lemmas(analyzer.articles[lexical_winner])))[0]},
    }
    if probe := PROBES.get(query["query_id"]):
        text, tsquery = (probe["value"], None) if probe["kind"] == "query" else (query["query"], probe["value"])
        after = analyzer.rankings(text, tsquery)
        after_ndcg = ndcg_at_k(after["hybrid"][:TOP_ARTICLES], judgments, 10)
        case["probe"] = {
            "kind": probe["kind"],
            "value": probe["value"],
            "positions": positions(after, judgments),
            "ndcg@10": after_ndcg,
            "effect": probe_effect(ndcg, after_ndcg),
        }
    return case


def analyze() -> dict:
    queries = {q["query_id"]: q for q in load_jsonl(QUERIES_FILE)}
    qrels = load_qrels()
    cases, meta = [], {}
    for chunking in CHUNKINGS:
        saved = json.loads((RESULTS_DIR / f"{SOURCE_SYSTEM}_{chunking}.json").read_text(encoding="utf-8"))
        meta[chunking] = {"chunks_sha256": saved["chunks_sha256"], "golden_sha256": saved["golden_sha256"]}
        analyzer = Analyzer(chunking)
        cases += [analyze_case(analyzer, queries[q], qrels[q], saved["per_query"][q]) for q in select_bad(saved["per_query"])]
    by_type = defaultdict(int)
    for case in cases:
        by_type[case["type"]] += 1
    return {"system": SOURCE_SYSTEM, "bad_ndcg": BAD_NDCG, "effect_strong": EFFECT_STRONG, "inputs": meta, "n_cases": len(cases), "cases_by_type": dict(by_type), "cases": cases}


def run() -> int:
    try:
        if problems := check_frozen():
            raise RuntimeError("; ".join(problems))
        result = analyze()
    except (RuntimeError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    RESULTS_FILE.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    for case in result["cases"]:
        probe = case.get("probe")
        extra = f" -> sonda {probe['ndcg@10']:.3f} ({probe['effect']})" if probe else " (sin sonda)"
        print(f"{case['query_id']} {case['chunking']:7} nDCG@10 {case['ndcg@10']:.3f}{extra}  {case['query'][:60]}")
    print(f"\n{result['n_cases']} casos guardados en {RESULTS_FILE.relative_to(ROOT)}")
    return 0
