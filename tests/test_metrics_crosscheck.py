"""Contraste independiente del nDCG con scikit-learn (`sklearn.metrics.ndcg_score`).

scikit-learn recibe puntuaciones, no listas ordenadas, y usa los valores de y_true como ganancias. Se
traduce cada ranking a puntuaciones estrictamente decrecientes (sin empates) y los artículos juzgados que el
ranking no devuelve reciben puntuaciones menores que todos los devueltos. Los rankings tienen al menos 10
elementos, así que esos artículos quedan fuera del top-K en ambas implementaciones.
"""

import random

import pytest

from scripts.metrics import load_qrels, ndcg_at_k

np = pytest.importorskip("numpy")
sklearn_metrics = pytest.importorskip("sklearn.metrics")

KS = (1, 3, 5, 10)
TRIALS_PER_QUERY = 25


def simulated_ranking(judgments, rng):
    """Mezcla aleatoria de una parte de los artículos juzgados y de distractores, con al menos 10 elementos."""
    judged = list(judgments)
    returned = [a for a in judged if rng.random() < 0.7]
    distractors = [("DISTRACTOR", i) for i in range(rng.randint(0, 12))]
    ranking = returned + distractors
    while len(ranking) < 10:
        ranking.append(("DISTRACTOR", 1000 + len(ranking)))
    rng.shuffle(ranking)
    return ranking


def sklearn_ndcg(ranking, judgments, k, exponential=False):
    unreturned = [a for a in judgments if a not in set(ranking)]
    items = ranking + unreturned
    grades = np.array([[judgments.get(a, 0) for a in items]], dtype=float)
    if exponential:
        grades = 2**grades - 1
    scores = np.array([[len(items) - position for position in range(len(items))]], dtype=float)  # estrictamente decreciente
    return sklearn_metrics.ndcg_score(grades, scores, k=k, ignore_ties=True)


def test_hand_example_matches_sklearn():
    judgments = {("C", 1): 3, ("C", 2): 2, ("C", 3): 1}
    ranking = [("C", 90), ("C", 2), ("C", 1), ("C", 91), ("C", 3)] + [("C", 100 + i) for i in range(5)]
    assert sklearn_ndcg(ranking, judgments, 5) == pytest.approx(0.6612, abs=1e-4)
    assert ndcg_at_k(ranking, judgments, 5) == pytest.approx(sklearn_ndcg(ranking, judgments, 5), abs=1e-12)


@pytest.mark.parametrize("exponential", [False, True], ids=["linear", "exponential"])
def test_ndcg_matches_sklearn_on_the_frozen_golden_set(exponential):
    rng = random.Random(2026)
    gain = "exponential" if exponential else "linear"
    comparisons = 0
    for judgments in load_qrels().values():
        for _ in range(TRIALS_PER_QUERY):
            ranking = simulated_ranking(judgments, rng)
            for k in KS:
                ours = ndcg_at_k(ranking, judgments, k, gain=gain)
                theirs = sklearn_ndcg(ranking, judgments, k, exponential)
                assert ours == pytest.approx(theirs, abs=1e-9)
                comparisons += 1
    assert comparisons == 40 * TRIALS_PER_QUERY * len(KS)
