import json

import pytest

from scripts import evaluate as ev


class FakeIndex:
    """Devuelve chunks con artículos repetidos, como haría un índice de ventanas."""

    def __init__(self, ranking):
        self.ranking = ranking

    def search(self, query, depth):
        return self.ranking[:depth]


def test_run_system_converts_chunks_to_distinct_articles():
    a, b, c = ("C", 1), ("C", 2), ("C", 3)
    index = FakeIndex([a, a, b, a, c, b])
    run = ev.run_system(index, [{"query_id": "q1", "query": "x"}], depth=10, top=2)
    assert run == {"q1": [a, b]}


def test_run_system_depth_limits_chunks_before_deduplication():
    a, b = ("C", 1), ("C", 2)
    run = ev.run_system(FakeIndex([a, a, a, b]), [{"query_id": "q1", "query": "x"}], depth=3, top=5)
    assert run == {"q1": [a]}  # b queda fuera de los 3 primeros chunks


def test_summarize_groups_by_query_type():
    queries = [{"query_id": "q1", "type": "paraphrase"}, {"query_id": "q2", "type": "exact_term"}, {"query_id": "q3", "type": "exact_term"}]
    per_query = {"q1": {"m": 1.0}, "q2": {"m": 0.0}, "q3": {"m": 0.5}}
    summary = ev.summarize(per_query, queries)
    assert summary["overall"]["n_queries"] == 3 and summary["overall"]["mean"]["m"] == pytest.approx(0.5)
    assert summary["paraphrase"]["mean"]["m"] == 1.0
    assert summary["exact_term"]["n_queries"] == 2 and summary["exact_term"]["mean"]["m"] == pytest.approx(0.25)
    assert "multi_article" not in summary  # sin consultas de ese tipo, no hay grupo


def test_mean_ci_of_constant_scores_is_degenerate():
    mean, low, high = ev.mean_ci({"q1": 0.4, "q2": 0.4, "q3": 0.4})
    assert mean == pytest.approx(0.4) and low == pytest.approx(0.4) and high == pytest.approx(0.4)


def test_evaluate_refuses_when_golden_is_not_frozen(monkeypatch):
    monkeypatch.setattr(ev, "read_hashes", lambda path: {})
    with pytest.raises(RuntimeError, match="congelado"):
        ev.evaluate("bm25", "article")


def test_evaluate_refuses_when_golden_was_modified(monkeypatch):
    monkeypatch.setattr(ev, "read_hashes", lambda path: {"qrels.jsonl": "x"})
    monkeypatch.setattr(ev, "check_frozen", lambda: ["qrels.jsonl: modificado después de congelar"])
    with pytest.raises(RuntimeError, match="modificado"):
        ev.evaluate("bm25", "article")


def test_evaluate_writes_a_reproducible_report(tmp_path):
    first = ev.evaluate("bm25", "article", results_dir=tmp_path)
    saved = json.loads((tmp_path / "bm25_article.json").read_text())
    assert saved["system"] == "bm25" and saved["chunking"] == "article"
    assert set(saved["per_query"]) == {f"q{i:03d}" for i in range(1, 41)}
    assert saved["summary"]["overall"]["n_queries"] == 40
    ev.evaluate("bm25", "article", results_dir=tmp_path)
    assert json.loads((tmp_path / "bm25_article.json").read_text()) == saved == json.loads(json.dumps(first))
