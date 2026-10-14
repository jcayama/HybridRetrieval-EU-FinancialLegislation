import pytest

from scripts.compare import compare, format_comparison

TYPES = {"q1": "paraphrase", "q2": "paraphrase", "q3": "exact_term", "q4": "multi_article"}


def report(values, golden="g"):
    return {"golden_sha256": {"qrels.jsonl": golden}, "per_query": {q: {"ndcg@10": v} for q, v in values.items()}}


def test_compare_counts_wins_losses_and_ties():
    a = report({"q1": 1.0, "q2": 0.5, "q3": 0.2, "q4": 0.4})
    b = report({"q1": 0.5, "q2": 0.5, "q3": 0.4, "q4": 0.4})
    result = compare(a, b, "ndcg@10", TYPES)
    assert result["overall"]["n"] == 4
    assert (result["overall"]["a_better"], result["overall"]["b_better"], result["overall"]["ties"]) == (1, 1, 2)
    assert result["overall"]["mean_diff"] == pytest.approx((0.5 + 0 - 0.2 + 0) / 4)
    assert result["paraphrase"]["n"] == 2 and result["exact_term"]["mean_diff"] == pytest.approx(-0.2)


def test_compare_refuses_different_golden_sets_or_queries():
    a = report({"q1": 1.0, "q2": 1.0, "q3": 1.0, "q4": 1.0})
    with pytest.raises(ValueError, match="golden"):
        compare(a, report({"q1": 1.0, "q2": 1.0, "q3": 1.0, "q4": 1.0}, golden="otro"), "ndcg@10", TYPES)
    with pytest.raises(ValueError, match="consultas"):
        compare(a, report({"q1": 1.0}), "ndcg@10", TYPES)


def test_format_comparison_mentions_both_systems():
    a = report({"q1": 1.0, "q2": 1.0, "q3": 1.0, "q4": 1.0})
    text = format_comparison("sysA", "sysB", "ndcg@10", compare(a, a, "ndcg@10", TYPES))
    assert "sysA − sysB" in text and "dif=+0.000" in text
