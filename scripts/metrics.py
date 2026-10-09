"""Métricas de evaluación de recuperación: Recall@K, MRR@K, nDCG@K y bootstrap pareado.

Funciones puras: no saben nada de bases de datos ni de modelos. Trabajan con *artículos*, identificados
por (celex, número). Un ranking es una lista de artículos ordenada de mejor a peor; los juicios de una
consulta son un dict {artículo: grado} con grados 1–3 (lo no juzgado es 0).

Decisiones (fijadas antes de ejecutar ninguna recuperación):
- «Relevante» para Recall y MRR = grado >= 2 (el grado 1 solo cuenta en nDCG).
- nDCG con ganancia lineal (3, 2, 1) por defecto; la exponencial (7, 3, 1) queda como análisis de sensibilidad.
- Los rankings de chunks se convierten a artículos con `dedupe_to_articles` (mejor posición de cada uno).
- Bootstrap pareado sobre consultas, semilla fija, intervalo percentil.
"""

import random
from dataclasses import dataclass
from math import log2
from pathlib import Path

from scripts.golden import QRELS_FILE, load_jsonl

Article = tuple[str, int]  # (celex, número de artículo)
Judgments = dict[Article, int]  # juicios de una consulta
Qrels = dict[str, Judgments]  # query_id -> juicios
Run = dict[str, list[Article]]  # query_id -> ranking de artículos

RELEVANT_THRESHOLD = 2
GAINS = {"linear": lambda grade: grade, "exponential": lambda grade: 2**grade - 1}


def load_qrels(path: Path = QRELS_FILE) -> Qrels:
    qrels: Qrels = {}
    for row in load_jsonl(path):
        qrels.setdefault(row["query_id"], {})[(row["celex"], row["article"])] = row["grade"]
    return qrels


def dedupe_to_articles(chunks: list[Article]) -> list[Article]:
    """Convierte un ranking de chunks (un artículo puede repetirse) en un ranking de artículos distintos,
    conservando la primera aparición, es decir, la mejor posición."""
    return list(dict.fromkeys(chunks))


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k debe ser positivo (recibido {k})")


def _relevant(judgments: Judgments, threshold: int) -> set[Article]:
    relevant = {article for article, grade in judgments.items() if grade >= threshold}
    if not relevant:
        raise ValueError(f"la consulta no tiene ningún artículo relevante (grado >= {threshold})")
    return relevant


def recall_at_k(ranking: list[Article], judgments: Judgments, k: int, threshold: int = RELEVANT_THRESHOLD) -> float:
    """Fracción de los artículos relevantes que aparecen en los K primeros del ranking."""
    _check_k(k)
    relevant = _relevant(judgments, threshold)
    top = dedupe_to_articles(ranking)[:k]
    return len(relevant.intersection(top)) / len(relevant)


def reciprocal_rank(ranking: list[Article], judgments: Judgments, k: int = 10, threshold: int = RELEVANT_THRESHOLD) -> float:
    """1 / posición del primer artículo relevante dentro de los K primeros; 0 si no hay ninguno."""
    _check_k(k)
    relevant = _relevant(judgments, threshold)
    for position, article in enumerate(dedupe_to_articles(ranking)[:k], start=1):
        if article in relevant:
            return 1 / position
    return 0.0


def _dcg(grades: list[int], gain) -> float:
    return sum(gain(grade) / log2(position + 1) for position, grade in enumerate(grades, start=1))


def ndcg_at_k(ranking: list[Article], judgments: Judgments, k: int = 10, gain: str = "linear") -> float:
    """nDCG@K con los grados 1–3. El ranking ideal se construye con todos los artículos juzgados,
    estén o no en el ranking devuelto."""
    if gain not in GAINS:
        raise ValueError(f"gain desconocida: {gain!r} (usa {', '.join(GAINS)})")
    _check_k(k)
    if not judgments:
        raise ValueError("la consulta no tiene juicios")
    gain_fn = GAINS[gain]
    obtained = [judgments.get(article, 0) for article in dedupe_to_articles(ranking)[:k]]
    ideal = sorted(judgments.values(), reverse=True)[:k]
    return _dcg(obtained, gain_fn) / _dcg(ideal, gain_fn)


def evaluate_run(
    run: Run,
    qrels: Qrels,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    ndcg_k: int = 10,
    mrr_k: int = 10,
    threshold: int = RELEVANT_THRESHOLD,
    gain: str = "linear",
) -> dict[str, dict[str, float]]:
    """Métricas por consulta. Exige que el run cubra exactamente las consultas de los juicios."""
    missing, unknown = sorted(set(qrels) - set(run)), sorted(set(run) - set(qrels))
    if missing or unknown:
        raise ValueError(f"el run no coincide con los juicios; faltan: {missing}; sobran: {unknown}")
    results: dict[str, dict[str, float]] = {}
    for query_id, judgments in qrels.items():
        ranking = run[query_id]
        metrics = {f"recall@{k}": recall_at_k(ranking, judgments, k, threshold) for k in ks}
        metrics[f"ndcg@{ndcg_k}"] = ndcg_at_k(ranking, judgments, ndcg_k, gain)
        metrics[f"mrr@{mrr_k}"] = reciprocal_rank(ranking, judgments, mrr_k, threshold)
        results[query_id] = metrics
    return results


def mean_metrics(per_query: dict[str, dict[str, float]]) -> dict[str, float]:
    names = next(iter(per_query.values()))
    return {name: sum(m[name] for m in per_query.values()) / len(per_query) for name in names}


@dataclass(frozen=True)
class BootstrapResult:
    mean_diff: float
    low: float
    high: float
    n_resamples: int
    seed: int
    confidence: float


def paired_bootstrap(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    n_resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Diferencia media A − B con intervalo percentil, remuestreando consultas con reposición.
    La misma remuestra se aplica a los dos sistemas (de ahí «pareado»)."""
    if set(scores_a) != set(scores_b):
        raise ValueError("los dos sistemas deben tener las mismas consultas")
    if not scores_a:
        raise ValueError("no hay consultas")
    diffs = [scores_a[q] - scores_b[q] for q in sorted(scores_a)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(n_resamples))
    alpha = 1 - confidence
    low = means[int(n_resamples * alpha / 2)]
    high = means[int(n_resamples * (1 - alpha / 2)) - 1]
    return BootstrapResult(sum(diffs) / n, low, high, n_resamples, seed, confidence)
