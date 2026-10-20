"""Distribución del coseno del mejor chunk: consultas del golden set (con respuesta) frente a consultas sin respuesta.

Objetivo: saber si un umbral sobre el coseno permitiría que el recuperador se abstuviera ante consultas fuera de
dominio, y a qué precio. Decisiones fijadas ANTES de calcular (no se ajustan viendo los resultados):
- Puntuación por consulta: coseno (1 − distancia `<=>`) del mejor chunk de la búsqueda densa exacta, con cada chunking.
- Con respuesta: las 40 consultas del golden set (todas tienen artículos relevantes). Sin respuesta: las 20 de
  data/abstention/queries.jsonl (10 fuera de tema y 10 «casi del dominio»), congeladas con hash. Que no tienen
  respuesta en el corpus lo afirma quien pone los juicios (Jorge), no un script.
- Resumen: mediana, mínimo y máximo por grupo; AUC (probabilidad de que una consulta con respuesta tenga más coseno que
  una sin respuesta, empates = 0,5) con IC por bootstrap (10.000 remuestras, semilla fija, remuestreando cada grupo).
- Tabla umbral → % del golden set que se conservaría / % de consultas sin respuesta que se rechazaría, en una rejilla
  fija (THRESHOLDS). No se elige un umbral automáticamente.
- Regla para PROPONER un umbral: el más alto que conserve al menos RETENTION (95 %) de las consultas del golden set;
  se informa cuántas consultas sin respuesta rechaza. Se prefiere rechazar de menos que callar respuestas válidas.
Límites: las consultas del golden set se escribieron «desde el artículo», así que su coseno es más alto que el de un
usuario real; y el umbral se calibra con las mismas consultas con las que se evalúa (optimista). Son 40 y 20 consultas:
los intervalos son anchos y el resultado es un indicio, no un umbral definitivo.
"""

import json
import math
import re
import statistics
import sys

import numpy as np

from schemas.retrieval import MAX_QUERY_CHARS
from scripts import db, embeddings
from scripts.dense import DenseIndex
from scripts.download_corpus import ROOT, read_hashes, sha256_bytes, write_hashes
from scripts.evaluate import CHUNK_FILES, RESULTS_DIR
from scripts.golden import QUERIES_FILE as GOLDEN_QUERIES_FILE
from scripts.golden import check_frozen as golden_check_frozen
from scripts.golden import load_jsonl

ABSTENTION_DIR = ROOT / "data" / "abstention"
QUERIES_FILE = embeddings.ABSTENTION_QUERIES_FILE
HASH_FILE = ABSTENTION_DIR / "abstention.sha256"
RESULTS_FILE = RESULTS_DIR / "abstention.json"
THRESHOLD_FILE = ABSTENTION_DIR / "threshold.json"  # lo que lee la API; se escribe con `python cli.py abstention --write-threshold`
GROUPS = ("off_topic", "near_domain")
QUOTA = 10  # consultas por grupo
QUERY_KEYS = {"query_id", "group", "query"}
CHUNKINGS = ("article", "window")
# La API solo se abstiene con ventanas: con artículos la separación es peor (el mayor coseno fuera de tema supera al mínimo del
# golden set) y el umbral rechazaba consultas legítimas («social scoring», «deep fake»). El análisis sigue midiendo ambos.
ABSTENTION_CHUNKINGS = ("window",)
RETENTION = 0.95
THRESHOLDS = tuple(round(0.40 + 0.02 * i, 2) for i in range(26))  # 0,40 … 0,90
N_RESAMPLES = 10_000
SEED = 0
TOP_SQL = """
SELECT chunk_id, celex, article, 1 - (embedding <=> %(vector)s::vector) AS cosine
FROM chunks
WHERE strategy = %(strategy)s AND embedding IS NOT NULL AND tenant_id IS NULL
ORDER BY embedding <=> %(vector)s::vector, chunk_id
LIMIT 1
"""


# ---------- funciones puras ----------


def validate(queries: list[dict], golden_queries: list[dict]) -> list[str]:
    """Errores del fichero de consultas sin respuesta (lista vacía si es válido)."""
    errors = []
    for row in queries:
        if set(row) != QUERY_KEYS:
            errors.append(f"{row.get('query_id', '?')}: claves {sorted(row)} en lugar de {sorted(QUERY_KEYS)}")
    ids = [row.get("query_id") for row in queries]
    if len(set(ids)) != len(ids):
        errors.append("query_id repetidos")
    for row in queries:
        if not re.fullmatch(r"n\d{3}", str(row.get("query_id", ""))):
            errors.append(f"{row.get('query_id')}: el query_id debe ser n seguido de tres cifras")
        if row.get("group") not in GROUPS:
            errors.append(f"{row.get('query_id')}: grupo {row.get('group')!r} no válido")
        text = str(row.get("query", "")).strip()
        if not text or len(text) > MAX_QUERY_CHARS:
            errors.append(f"{row.get('query_id')}: la consulta está vacía o supera {MAX_QUERY_CHARS} caracteres")
    for group in GROUPS:
        found = sum(1 for row in queries if row.get("group") == group)
        if found != QUOTA:
            errors.append(f"grupo {group}: {found} consultas en lugar de {QUOTA}")
    texts = [str(row.get("query", "")).strip().lower() for row in queries]
    if len(set(texts)) != len(texts):
        errors.append("consultas repetidas")
    if overlap := sorted(set(texts) & {row["query"].strip().lower() for row in golden_queries}):
        errors.append(f"consultas que ya están en el golden set: {overlap}")
    return errors


def auc(positives: list[float], negatives: list[float]) -> float:
    """P(positivo > negativo) + 0,5 · P(empate): la probabilidad de que una consulta con respuesta puntúe más."""
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("hacen falta puntuaciones de los dos grupos")
    pos, neg = np.asarray(positives, dtype=float)[:, None], np.asarray(negatives, dtype=float)[None, :]
    return float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / (pos.shape[0] * neg.shape[1]))


def bootstrap_auc(positives: list[float], negatives: list[float], n_resamples: int = N_RESAMPLES, seed: int = SEED, confidence: float = 0.95) -> tuple[float, float]:
    """IC percentil del AUC, remuestreando con reposición cada grupo por separado."""
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(positives, dtype=float), np.asarray(negatives, dtype=float)
    values = np.empty(n_resamples)
    for i in range(n_resamples):
        values[i] = auc(rng.choice(pos, size=pos.size), rng.choice(neg, size=neg.size))
    values.sort()
    alpha = 1 - confidence
    return float(values[int(n_resamples * alpha / 2)]), float(values[int(n_resamples * (1 - alpha / 2)) - 1])


def threshold_for_retention(scores: list[float], retention: float = RETENTION) -> float:
    """El umbral más alto (un valor observado) con el que se conserva (puntuación >= umbral) al menos `retention` de las consultas."""
    if not scores:
        raise ValueError("no hay puntuaciones")
    needed = math.ceil(retention * len(scores) - 1e-9)
    return sorted(scores, reverse=True)[needed - 1]


def share_at_least(scores: list[float], threshold: float) -> tuple[int, int]:
    """(consultas con puntuación >= umbral, total)."""
    return sum(1 for s in scores if s >= threshold), len(scores)


def summary(scores: list[float]) -> dict[str, float]:
    return {"min": min(scores), "median": statistics.median(scores), "max": max(scores)}


def threshold_table(golden: list[float], negatives_by_group: dict[str, list[float]], thresholds=THRESHOLDS) -> list[dict]:
    """Por umbral: golden conservado (puntuación >= umbral) y consultas sin respuesta rechazadas (puntuación < umbral)."""
    rows = []
    for t in thresholds:
        kept, total = share_at_least(golden, t)
        row = {"threshold": t, "golden_kept": kept, "golden_total": total}
        for group, scores in negatives_by_group.items():
            row[f"{group}_rejected"] = len(scores) - share_at_least(scores, t)[0]
            row[f"{group}_total"] = len(scores)
        rows.append(row)
    return rows


# ---------- con Postgres y los vectores precalculados ----------


def top_chunks(chunking: str, vectors: np.ndarray) -> list[dict]:
    """Mejor chunk (búsqueda densa exacta, corpus público) de cada vector: chunk_id, artículo y coseno."""
    DenseIndex(chunking, sha256_bytes(CHUNK_FILES[chunking].read_bytes()), lambda q: None)  # se niega si la base está desfasada
    out = []
    with db.connect() as conn:
        for vector in vectors:
            chunk_id, celex, article, cosine = conn.execute(TOP_SQL, {"vector": db.to_vector_literal(vector), "strategy": chunking}).fetchone()
            out.append({"chunk_id": chunk_id, "celex": celex, "article": article, "cosine": float(cosine)})
    return out


def analyze() -> dict:
    golden = load_jsonl(GOLDEN_QUERIES_FILE)
    negatives = load_jsonl(QUERIES_FILE)
    golden_ids, golden_vectors = embeddings.load_embeddings("queries")
    negative_ids, negative_vectors = embeddings.load_embeddings("abstention_queries")
    if golden_ids != [q["query_id"] for q in golden] or negative_ids != [q["query_id"] for q in negatives]:
        raise RuntimeError("los embeddings no corresponden a las consultas actuales; ejecuta `python cli.py embed`")
    result = {
        "inputs": {
            "golden_sha256": read_hashes(ROOT / "data" / "golden" / "golden.sha256"),
            "abstention_sha256": read_hashes(HASH_FILE),
            "embeddings_sha256": {name: embeddings.read_manifest()["files"][name]["sha256"] for name in ("queries", "abstention_queries", "chunks_article", "chunks_window")},
        },
        "method": {"retention": RETENTION, "thresholds": list(THRESHOLDS), "n_resamples": N_RESAMPLES, "seed": SEED, "score": "cosine of the best chunk (exact dense search)"},
        "chunkings": {},
    }
    for chunking in CHUNKINGS:
        pos_rows = [{"query_id": q["query_id"], "type": q["type"], "query": q["query"]} | top for q, top in zip(golden, top_chunks(chunking, golden_vectors), strict=True)]
        neg_rows = [{"query_id": q["query_id"], "group": q["group"], "query": q["query"]} | top for q, top in zip(negatives, top_chunks(chunking, negative_vectors), strict=True)]
        pos = [r["cosine"] for r in pos_rows]
        by_group = {g: [r["cosine"] for r in neg_rows if r["group"] == g] for g in GROUPS}
        all_neg = [r["cosine"] for r in neg_rows]
        proposal_t = threshold_for_retention(pos)
        result["chunkings"][chunking] = {
            "golden": pos_rows,
            "abstention": neg_rows,
            "summary": {"golden": summary(pos)} | {g: summary(s) for g, s in by_group.items()},
            "auc": {"all": _auc_block(pos, all_neg)} | {g: _auc_block(pos, s) for g, s in by_group.items()},
            "table": threshold_table(pos, by_group),
            "proposal": {
                "threshold": proposal_t,
                "golden_kept": share_at_least(pos, proposal_t)[0],
                "golden_total": len(pos),
                "rejected": {g: len(s) - share_at_least(s, proposal_t)[0] for g, s in by_group.items()},
                "golden_rejected": [r["query_id"] for r in pos_rows if r["cosine"] < proposal_t],
            },
        }
    return result


def _auc_block(pos: list[float], neg: list[float]) -> dict:
    low, high = bootstrap_auc(pos, neg)
    return {"value": auc(pos, neg), "ci_low": low, "ci_high": high}


# ---------- umbral que usa la API ----------


def floor_to_thousandths(value: float) -> float:
    """Hacia abajo a milésimas: el modelo puede diferir en el último decimal entre plataformas, y así la consulta que
    fijó el umbral no queda en el filo. Solo hace el umbral más permisivo en menos de 0,001."""
    return math.floor(value * 1000 + 1e-9) / 1000


def threshold_spec(result: dict) -> dict:
    """Contenido de threshold.json a partir de results/abstention.json."""
    return {
        "use": "Abstención de la API (solo con chunking=window): si el coseno del mejor chunk es menor que el umbral, no se devuelven resultados. Solo descarta consultas claramente ajenas al corpus.",
        "rule": f"el coseno más alto que conserva al menos el {RETENTION:.0%} de las consultas del golden set, redondeado hacia abajo a milésimas",
        "retention": RETENTION,
        "thresholds": {chunking: floor_to_thousandths(result["chunkings"][chunking]["proposal"]["threshold"]) for chunking in ABSTENTION_CHUNKINGS},
        "calibrated_on": result["inputs"],
    }


def load_thresholds(path=THRESHOLD_FILE) -> dict[str, float]:
    """Umbrales por chunking, o {} si no hay fichero (sin abstención). Falla ante un fichero mal formado."""
    if not path.exists():
        return {}
    thresholds = json.loads(path.read_text(encoding="utf-8"))["thresholds"]
    if set(thresholds) != set(ABSTENTION_CHUNKINGS) or not all(isinstance(v, int | float) and 0 < v < 1 for v in thresholds.values()):
        raise RuntimeError(f"{path.name}: se esperan umbrales entre 0 y 1 para {', '.join(ABSTENTION_CHUNKINGS)}")
    return {k: float(v) for k, v in thresholds.items()}


# ---------- congelado y ejecución ----------


def hashes_of_files() -> dict[str, str]:
    return {QUERIES_FILE.name: sha256_bytes(QUERIES_FILE.read_bytes())}


def check_frozen() -> list[str]:
    frozen = read_hashes(HASH_FILE)
    if not frozen:
        return ["las consultas sin respuesta no están congeladas; ejecuta `python cli.py freeze-abstention`"]
    current = hashes_of_files()
    return [f"{name}: modificado después de congelar (SHA-256 distinto de abstention.sha256)" for name, digest in frozen.items() if current.get(name) != digest]


def run_freeze() -> int:
    try:
        errors = validate(load_jsonl(QUERIES_FILE), load_jsonl(GOLDEN_QUERIES_FILE))
    except (ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if errors:
        print("ERROR: hay errores de validación; no se congela:\n" + "\n".join(f"  - {e}" for e in errors), file=sys.stderr)
        return 1
    write_hashes(hashes_of_files(), HASH_FILE)
    print(f"Consultas sin respuesta congeladas en {HASH_FILE.relative_to(ROOT)}")
    return 0


def run(write_threshold: bool = False) -> int:
    try:
        if problems := golden_check_frozen() + check_frozen():
            raise RuntimeError("; ".join(problems))
        result = analyze()
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    RESULTS_FILE.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    if write_threshold:
        THRESHOLD_FILE.write_text(json.dumps(threshold_spec(result), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Umbrales escritos en {THRESHOLD_FILE.relative_to(ROOT)}: {load_thresholds()}")
    for chunking, data in result["chunkings"].items():
        s, a, p = data["summary"], data["auc"]["all"], data["proposal"]
        print(f"[{chunking}] coseno mediano: golden {s['golden']['median']:.3f} | fuera de tema {s['off_topic']['median']:.3f} | casi del dominio {s['near_domain']['median']:.3f}")
        print(f"          AUC {a['value']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}] | umbral propuesto {p['threshold']:.3f}: conserva {p['golden_kept']}/{p['golden_total']}, rechaza {p['rejected']}")
    print(f"\nResultados guardados en {RESULTS_FILE.relative_to(ROOT)}")
    return 0
