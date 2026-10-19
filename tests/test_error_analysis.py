import json

import pytest

from scripts import error_analysis as ea
from scripts.evaluate import RESULTS_DIR
from scripts.golden import QUERIES_FILE, load_jsonl
from scripts.metrics import load_qrels
from tests.helpers import skip_or_fail

A, B, C = ("32024R1689", 1), ("32024R1689", 2), ("32024R1689", 3)


def test_a_query_is_bad_below_the_ndcg_threshold_or_with_nothing_relevant_in_the_top_10():
    assert ea.is_bad({"ndcg@10": 0.299, "recall@10": 1.0})
    assert not ea.is_bad({"ndcg@10": 0.3, "recall@10": 1.0})  # el umbral exacto no es «mala»
    assert ea.is_bad({"ndcg@10": 0.9, "recall@10": 0.0})  # ningún relevante, aunque el nDCG sea alto (p. ej. solo grado 1)
    assert not ea.is_bad({"ndcg@10": 0.31, "recall@10": 0.25})


def test_select_bad_returns_sorted_ids():
    per_query = {"q3": {"ndcg@10": 0.1, "recall@10": 0.0}, "q1": {"ndcg@10": 0.9, "recall@10": 1.0}, "q2": {"ndcg@10": 0.2, "recall@10": 0.5}}
    assert ea.select_bad(per_query) == ["q2", "q3"]


def test_position_is_one_based_and_none_when_absent():
    assert ea.position([A, B, C], B) == 2 and ea.position([A, B], C) is None
    assert ea.best_chunk_position([("c1", A), ("c2", B), ("c3", B)], B) == 2  # el mejor chunk del artículo, no el último


def test_lemma_split_keeps_the_query_order():
    assert ea.lemma_split(["stress", "test", "fine"], {"test", "fine", "other"}) == (["test", "fine"], ["stress"])


@pytest.mark.parametrize(
    ("before", "after", "effect"),
    [(0.5, 0.75, "explains"), (0.0, 0.1, "explains"), (0.5, 0.55, "partial"), (0.5, 0.5, "none"), (0.5, 0.4, "none")],
)
def test_probe_effect_labels(before, after, effect):
    assert ea.probe_effect(before, after) == effect


def test_hybrid_ranking_fuses_by_chunk_then_collapses_to_distinct_articles():
    # RRF (k=60): c1 y c3 empatan en 1/61, c2 y c4 en 1/62; el desempate por id da c1, c3, c2, c4 -> A, C, B, B -> A, C, B
    lexical = [("c1", A), ("c2", B)]
    semantic = [("c3", C), ("c4", B)]
    assert ea.hybrid_ranking(lexical, semantic) == [A, C, B]
    assert ea.hybrid_ranking([("c1", A), ("c2", B)], [("c2", B), ("c1", A)]) == [A, B]  # un chunk en las dos listas suma ambos aportes


def test_label_uses_the_act_name():
    assert ea.label(("32022R2554", 28)) == "DORA 28" and ea.label(("32024R1689", 99)) == "AI Act 99"


def test_probes_are_well_formed_and_refer_to_golden_queries():
    ids = {q["query_id"] for q in load_jsonl(QUERIES_FILE)}
    queries = {q["query_id"]: q["query"] for q in load_jsonl(QUERIES_FILE)}
    for query_id, probe in ea.PROBES.items():
        assert query_id in ids and probe["kind"] in {"query", "fts_tsquery"} and probe["value"].strip()
        assert probe["value"] != queries[query_id]  # una sonda cambia algo


# ---------- el fichero versionado frente a results/ ----------


@pytest.fixture(scope="module")
def saved():
    return json.loads(ea.RESULTS_FILE.read_text(encoding="utf-8"))


def test_the_saved_analysis_covers_exactly_the_bad_queries_of_results(saved):
    for chunking in ea.CHUNKINGS:
        result = json.loads((RESULTS_DIR / f"hybrid_{chunking}.json").read_text(encoding="utf-8"))
        cases = {c["query_id"]: c for c in saved["cases"] if c["chunking"] == chunking}
        assert sorted(cases) == ea.select_bad(result["per_query"]), "results/error_analysis.json está desfasado: ejecuta `python cli.py error-analysis`"
        assert saved["inputs"][chunking]["chunks_sha256"] == result["chunks_sha256"] and saved["inputs"][chunking]["golden_sha256"] == result["golden_sha256"]
        for query_id, case in cases.items():
            assert case["ndcg@10"] == pytest.approx(result["per_query"][query_id]["ndcg@10"], abs=1e-9)


def test_the_saved_probes_match_the_declared_ones_and_their_effect_label(saved):
    assert saved["n_cases"] == len(saved["cases"]) and saved["bad_ndcg"] == ea.BAD_NDCG and saved["effect_strong"] == ea.EFFECT_STRONG
    for case in saved["cases"]:
        probe = case.get("probe")
        assert (probe is not None) == (case["query_id"] in ea.PROBES), case["query_id"]
        if probe:
            declared = ea.PROBES[case["query_id"]]
            assert (probe["kind"], probe["value"]) == (declared["kind"], declared["value"])
            assert probe["effect"] == ea.probe_effect(case["ndcg@10"], probe["ndcg@10"])


def test_the_saved_positions_cover_every_judged_article(saved):
    qrels = load_qrels()
    for case in saved["cases"]:
        assert len(case["positions"]) == len(qrels[case["query_id"]])
        assert all(set(p) == {"grade", "fts", "dense", "hybrid"} for p in case["positions"].values())


# ---------- con Postgres ----------


def test_a_case_is_regenerated_from_the_database_and_reproduces_the_saved_metrics():
    """q036 con artículos: su sonda cambia solo la tsquery de FTS, así que no necesita el modelo (torch)."""
    queries = {q["query_id"]: q for q in load_jsonl(QUERIES_FILE)}
    result = json.loads((RESULTS_DIR / "hybrid_article.json").read_text(encoding="utf-8"))
    try:
        analyzer = ea.Analyzer("article")
    except Exception as error:  # sin Postgres o sin corpus cargado (en la CI es un fallo, no un salto)
        skip_or_fail(f"base de datos no disponible o sin cargar: {error}")
    case = ea.analyze_case(analyzer, queries["q036"], load_qrels()["q036"], result["per_query"]["q036"])  # lanza si no reproduce results/
    assert case["ndcg@10"] == pytest.approx(result["per_query"]["q036"]["ndcg@10"], abs=1e-9)
    assert case["probe"]["effect"] == "explains" and case["probe"]["positions"]["DORA 28"]["fts"] == 1
    assert case["lemma_chunk_frequency"]["inform"] > case["lemma_chunk_frequency"]["regist"]


def test_analyze_case_refuses_metrics_that_the_regenerated_ranking_does_not_reproduce():
    queries = {q["query_id"]: q for q in load_jsonl(QUERIES_FILE)}
    try:
        analyzer = ea.Analyzer("article")
    except Exception as error:
        skip_or_fail(f"base de datos no disponible o sin cargar: {error}")
    with pytest.raises(RuntimeError, match="no reproduce"):
        ea.analyze_case(analyzer, queries["q036"], load_qrels()["q036"], {"ndcg@10": 0.5, "recall@10": 1.0})
    real = json.loads((RESULTS_DIR / "hybrid_article.json").read_text(encoding="utf-8"))["per_query"]["q036"]
    with pytest.raises(RuntimeError, match="no reproduce"):  # el nDCG coincide, el recall no
        ea.analyze_case(analyzer, queries["q036"], load_qrels()["q036"], {"ndcg@10": real["ndcg@10"], "recall@10": 1.0})
