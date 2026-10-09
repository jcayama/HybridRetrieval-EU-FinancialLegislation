"""Tests del evaluador con un ejemplo hecho a mano.

Ejemplo base: juicios de una consulta A=3, B=2, C=1 y ranking devuelto X, B, A, Y, C
(X e Y son irrelevantes). Relevante = grado >= 2, es decir {A, B}.
Las cifras esperadas se calcularon a mano (ver README, sección Evaluación).
"""

import pytest

from scripts.metrics import (
    dedupe_to_articles,
    evaluate_run,
    load_qrels,
    mean_metrics,
    ndcg_at_k,
    paired_bootstrap,
    recall_at_k,
    reciprocal_rank,
)

C = "C1"
A, B, CC, X, Y = (C, 1), (C, 2), (C, 3), (C, 90), (C, 91)
QRELS = {A: 3, B: 2, CC: 1}
RANKING = [X, B, A, Y, CC]


# ---------- ventanas -> artículos ----------


def test_dedupe_keeps_best_position_of_each_article():
    assert dedupe_to_articles([A, A, B, CC]) == [A, B, CC]
    assert dedupe_to_articles([B, A, B, A]) == [B, A]


def test_dedupe_empty():
    assert dedupe_to_articles([]) == []


def test_metrics_deduplicate_repeated_articles_internally():
    # ventanas del mismo artículo no deben ocupar posiciones del top-K ni inflar el DCG
    repeated = [X, X, B, B, A, A]
    clean = [X, B, A]
    assert recall_at_k(repeated, QRELS, 3) == pytest.approx(recall_at_k(clean, QRELS, 3)) == pytest.approx(1.0)
    assert reciprocal_rank(repeated, QRELS) == pytest.approx(1 / 2)  # B es el 2.º artículo distinto
    assert ndcg_at_k(repeated, QRELS, 10) == pytest.approx(ndcg_at_k(clean, QRELS, 10))


# ---------- Recall@K ----------


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 0.5), (3, 1.0), (5, 1.0), (10, 1.0)])
def test_recall_at_k_hand_example(k, expected):
    assert recall_at_k(RANKING, QRELS, k) == pytest.approx(expected)


def test_recall_threshold_changes_the_relevant_set():
    # con umbral 1, C también es relevante: en el top-3 hay A y B de {A, B, C}
    assert recall_at_k(RANKING, QRELS, 3, threshold=1) == pytest.approx(2 / 3)


def test_recall_empty_ranking_is_zero():
    assert recall_at_k([], QRELS, 5) == 0.0


def test_recall_fails_loudly_without_relevant_items():
    with pytest.raises(ValueError, match="relevante"):
        recall_at_k(RANKING, {CC: 1}, 3)  # nada con grado >= 2
    with pytest.raises(ValueError):
        recall_at_k(RANKING, {}, 3)


def test_recall_rejects_non_positive_k():
    with pytest.raises(ValueError):
        recall_at_k(RANKING, QRELS, 0)


# ---------- MRR ----------


def test_reciprocal_rank_hand_example():
    assert reciprocal_rank(RANKING, QRELS) == pytest.approx(0.5)


def test_reciprocal_rank_ignores_grade_1_items():
    # C (grado 1) no cuenta como relevante: el primero relevante es B, en la posición 3
    assert reciprocal_rank([CC, X, B], QRELS) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_beyond_cutoff():
    ranking = [(C, 100 + i) for i in range(10)] + [A]  # A en la posición 11
    assert reciprocal_rank(ranking, QRELS, k=10) == 0.0
    assert reciprocal_rank(ranking, QRELS, k=11) == pytest.approx(1 / 11)


def test_reciprocal_rank_first_position():
    assert reciprocal_rank([A, B], QRELS) == 1.0


# ---------- nDCG@K ----------


def test_ndcg_linear_hand_example():
    assert ndcg_at_k(RANKING, QRELS, 5) == pytest.approx(0.6612, abs=1e-4)
    assert ndcg_at_k(RANKING, QRELS, 3) == pytest.approx(0.58, abs=1e-4)


def test_ndcg_exponential_hand_example():
    assert ndcg_at_k(RANKING, QRELS, 5, gain="exponential") == pytest.approx(0.6153, abs=1e-4)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([A, B, CC], QRELS, 10) == pytest.approx(1.0)
    assert ndcg_at_k([A, B, CC, X, Y], QRELS, 10) == pytest.approx(1.0)


def test_ndcg_ideal_counts_judged_items_not_retrieved():
    # solo A en el ranking: DCG = 3; el ideal (A, B, C) @5 = 4.7619 -> 0.63
    assert ndcg_at_k([A], QRELS, 5) == pytest.approx(0.63, abs=1e-4)


def test_ndcg_irrelevant_only_is_zero_and_empty_is_zero():
    assert ndcg_at_k([X, Y], QRELS, 5) == 0.0
    assert ndcg_at_k([], QRELS, 5) == 0.0


def test_ndcg_worse_order_scores_lower():
    assert ndcg_at_k([B, A, CC], QRELS, 10) < ndcg_at_k([A, B, CC], QRELS, 10)


def test_ndcg_input_validation():
    with pytest.raises(ValueError, match="gain"):
        ndcg_at_k(RANKING, QRELS, 5, gain="cuadratica")
    with pytest.raises(ValueError):
        ndcg_at_k(RANKING, {}, 5)
    with pytest.raises(ValueError):
        ndcg_at_k(RANKING, QRELS, 0)


# ---------- evaluate_run ----------


def test_evaluate_run_returns_all_metrics_per_query():
    qrels = {"q1": QRELS, "q2": {A: 3}}
    run = {"q1": RANKING, "q2": [A]}
    per_query = evaluate_run(run, qrels)
    assert set(per_query) == {"q1", "q2"}
    assert set(per_query["q1"]) == {"recall@1", "recall@3", "recall@5", "recall@10", "ndcg@10", "mrr@10"}
    assert per_query["q1"]["recall@3"] == pytest.approx(1.0)
    assert per_query["q1"]["mrr@10"] == pytest.approx(0.5)
    assert per_query["q2"]["mrr@10"] == 1.0
    means = mean_metrics(per_query)
    assert means["mrr@10"] == pytest.approx(0.75)


def test_evaluate_run_rejects_missing_or_unknown_queries():
    qrels = {"q1": QRELS, "q2": {A: 3}}
    with pytest.raises(ValueError, match="q2"):
        evaluate_run({"q1": RANKING}, qrels)  # falta q2: no se rellena con ceros en silencio
    with pytest.raises(ValueError, match="q9"):
        evaluate_run({"q1": RANKING, "q2": [A], "q9": [A]}, qrels)


# ---------- bootstrap pareado ----------


def test_bootstrap_identical_systems():
    scores = {f"q{i}": i / 10 for i in range(10)}
    result = paired_bootstrap(scores, dict(scores), n_resamples=500)
    assert result.mean_diff == 0.0 and result.low == 0.0 and result.high == 0.0


def test_bootstrap_constant_improvement():
    a = {f"q{i}": 0.5 + 0.1 for i in range(10)}
    b = {f"q{i}": 0.5 for i in range(10)}
    result = paired_bootstrap(a, b, n_resamples=500)
    assert result.mean_diff == pytest.approx(0.1)
    assert result.low == pytest.approx(0.1) and result.high == pytest.approx(0.1)


def test_bootstrap_noise_interval_contains_zero():
    # diferencias +1 / -1 alternas: media 0 e intervalo con anchura
    a = {f"q{i}": 1.0 if i % 2 == 0 else 0.0 for i in range(20)}
    b = {f"q{i}": 0.0 if i % 2 == 0 else 1.0 for i in range(20)}
    result = paired_bootstrap(a, b, n_resamples=2000, seed=1)
    assert result.mean_diff == pytest.approx(0.0)
    assert result.low < 0 < result.high


def test_bootstrap_clear_win_has_positive_interval():
    a = {f"q{i}": 1.0 for i in range(30)}
    b = {f"q{i}": 0.0 if i < 25 else 1.0 for i in range(30)}
    result = paired_bootstrap(a, b, n_resamples=2000)
    assert result.low > 0


def test_bootstrap_is_reproducible_with_seed():
    a = {f"q{i}": (i * 7 % 10) / 10 for i in range(25)}
    b = {f"q{i}": (i * 3 % 10) / 10 for i in range(25)}
    first = paired_bootstrap(a, b, n_resamples=1000, seed=42)
    assert paired_bootstrap(a, b, n_resamples=1000, seed=42) == first
    assert paired_bootstrap(a, b, n_resamples=1000, seed=43) != first


def test_bootstrap_requires_same_queries():
    with pytest.raises(ValueError, match="consultas"):
        paired_bootstrap({"q1": 1.0}, {"q2": 1.0})


# ---------- sobre el golden set congelado ----------


def test_perfect_and_empty_runs_on_the_frozen_golden_set():
    qrels = load_qrels()
    assert len(qrels) == 40
    assert sum(len(v) for v in qrels.values()) == 102

    perfect = {q: sorted(j, key=lambda key: -j[key]) for q, j in qrels.items()}
    for q_metrics in evaluate_run(perfect, qrels).values():
        assert q_metrics["recall@10"] == pytest.approx(1.0)
        assert q_metrics["ndcg@10"] == pytest.approx(1.0)
        assert q_metrics["mrr@10"] == 1.0

    nothing = {q: [("NONE", i) for i in range(10)] for q in qrels}
    assert all(v == 0.0 for m in evaluate_run(nothing, qrels).values() for v in m.values())
