import json

import numpy as np
import pytest

from scripts import abstention as ab
from scripts import embeddings
from scripts.download_corpus import read_hashes
from scripts.golden import QUERIES_FILE as GOLDEN_QUERIES_FILE
from scripts.golden import load_jsonl
from tests.helpers import skip_or_fail


def rows(n_off=10, n_near=10):
    return [{"query_id": f"n{i:03d}", "group": "off_topic" if i <= n_off else "near_domain", "query": f"consulta {i}"} for i in range(1, n_off + n_near + 1)]


# ---------- validación del fichero ----------


def test_a_well_formed_file_has_no_errors():
    assert ab.validate(rows(), [{"query": "otra cosa"}]) == []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda r: r[0].update(extra=1), "claves"),
        (lambda r: r[1].update(query_id=r[0]["query_id"]), "repetidos"),
        (lambda r: r[0].update(query_id="x1"), "n seguido de tres cifras"),
        (lambda r: r[0].update(group="otro"), "no válido"),
        (lambda r: r[0].update(query="   "), "vacía"),
        (lambda r: r[0].update(query="x" * 501), "supera"),
        (lambda r: r[1].update(query=r[0]["query"].upper()), "repetidas"),
        (lambda r: r.pop(), "en lugar de 10"),
    ],
)
def test_each_kind_of_problem_in_the_file_is_reported(mutate, expected):
    data = rows()
    mutate(data)
    assert any(expected in error for error in ab.validate(data, [{"query": "otra cosa"}]))


def test_a_query_already_in_the_golden_set_is_rejected_ignoring_case():
    data = rows()
    data[0]["query"] = "Lead Overseer"
    assert any("golden" in error for error in ab.validate(data, [{"query": "lead overseer"}]))


# ---------- estadística, con ejemplos hechos a mano ----------


def test_auc_hand_example():
    # positivos 3 y 2, negativos 1 y 2: 3>1, 3>2, 2>1 ganan (3) y 2=2 empata (0,5) -> 3,5 / 4
    assert ab.auc([3, 2], [1, 2]) == pytest.approx(0.875)
    assert ab.auc([5, 6], [1, 2]) == 1.0 and ab.auc([1, 2], [5, 6]) == 0.0 and ab.auc([1, 1], [1, 1]) == 0.5


def test_auc_needs_both_groups():
    with pytest.raises(ValueError, match="dos grupos"):
        ab.auc([], [1.0])


def test_bootstrap_auc_is_deterministic_and_contains_the_estimate_in_a_noisy_case():
    pos = [0.5 + 0.017 * i for i in range(15)]
    neg = [0.4 + 0.019 * i for i in range(15)]  # se solapan, con muchos valores distintos del AUC posibles
    low, high = ab.bootstrap_auc(pos, neg, n_resamples=300)
    assert (low, high) == ab.bootstrap_auc(pos, neg, n_resamples=300)  # misma semilla, mismo intervalo
    assert (low, high) != ab.bootstrap_auc(pos, neg, n_resamples=300, seed=1)  # otra semilla, otro intervalo
    assert low <= ab.auc(pos, neg) <= high and 0.0 <= low < high <= 1.0


def test_bootstrap_auc_is_degenerate_when_the_groups_do_not_overlap():
    assert ab.bootstrap_auc([0.9, 0.8, 0.7], [0.3, 0.2, 0.1], n_resamples=500) == (1.0, 1.0)


@pytest.mark.parametrize(
    ("scores", "retention", "threshold", "kept"),
    [
        (list(range(1, 21)), 0.95, 2, 19),  # 19 de 20 = 95 %: la 19.ª mejor puntuación
        (list(range(1, 41)), 0.95, 3, 38),  # 38 de 40
        ([5, 5, 5, 1], 0.75, 5, 3),
        ([5, 4, 3, 2], 1.0, 2, 4),  # conservar todo: el mínimo
        ([5, 4, 3, 2], 0.25, 5, 1),
        (list(range(1, 11)), 0.95, 1, 10),  # 0,95 × 10 = 9,5: hay que redondear hacia arriba (9 de 10 sería solo el 90 %)
    ],
)
def test_threshold_for_retention_keeps_at_least_the_requested_share(scores, retention, threshold, kept):
    t = ab.threshold_for_retention(scores, retention)
    assert t == threshold and ab.share_at_least(scores, t)[0] == kept and kept / len(scores) >= retention


def test_the_threshold_itself_is_kept_and_the_next_value_down_is_not():
    assert ab.share_at_least([0.5, 0.6, 0.7], 0.6) == (2, 3)  # >= umbral


def test_threshold_for_retention_rejects_an_empty_list():
    with pytest.raises(ValueError):
        ab.threshold_for_retention([])


def test_summary_and_threshold_table_hand_example():
    assert ab.summary([0.4, 0.6, 0.8]) == {"min": 0.4, "median": 0.6, "max": 0.8}
    assert ab.summary([0.1, 0.2, 0.9])["median"] == 0.2  # la mediana, no la media (0,4)
    table = ab.threshold_table([0.5, 0.7, 0.9], {"off_topic": [0.2, 0.4], "near_domain": [0.6, 0.8]}, thresholds=(0.5, 0.7))
    assert table == [
        {"threshold": 0.5, "golden_kept": 3, "golden_total": 3, "off_topic_rejected": 2, "off_topic_total": 2, "near_domain_rejected": 0, "near_domain_total": 2},
        {"threshold": 0.7, "golden_kept": 2, "golden_total": 3, "off_topic_rejected": 2, "off_topic_total": 2, "near_domain_rejected": 1, "near_domain_total": 2},
    ]


# ---------- los datos versionados y el resultado guardado ----------


def test_the_versioned_queries_are_valid_and_frozen():
    assert ab.validate(load_jsonl(ab.QUERIES_FILE), load_jsonl(GOLDEN_QUERIES_FILE)) == []
    assert ab.check_frozen() == [], "las consultas sin respuesta cambiaron después de congelar"


@pytest.fixture(scope="module")
def saved():
    return json.loads(ab.RESULTS_FILE.read_text(encoding="utf-8"))


def test_the_saved_result_matches_the_current_inputs(saved):
    manifest = embeddings.read_manifest()["files"]
    assert saved["inputs"]["abstention_sha256"] == read_hashes(ab.HASH_FILE)
    assert saved["inputs"]["golden_sha256"] == read_hashes(ab.ROOT / "data" / "golden" / "golden.sha256")
    for name, digest in saved["inputs"]["embeddings_sha256"].items():
        assert manifest[name]["sha256"] == digest, "results/abstention.json está desfasado: ejecuta `python cli.py abstention`"


def test_the_saved_summaries_follow_from_the_saved_cosines(saved):
    for block in saved["chunkings"].values():
        pos = [r["cosine"] for r in block["golden"]]
        neg = [r["cosine"] for r in block["abstention"]]
        assert (len(pos), len(neg)) == (40, 20)
        assert block["auc"]["all"]["value"] == pytest.approx(ab.auc(pos, neg))
        threshold = ab.threshold_for_retention(pos)
        assert block["proposal"]["threshold"] == threshold and block["proposal"]["golden_kept"] == ab.share_at_least(pos, threshold)[0]
        assert block["proposal"]["golden_kept"] / len(pos) >= ab.RETENTION
        assert block["proposal"]["golden_rejected"] == [r["query_id"] for r in block["golden"] if r["cosine"] < threshold]
        for group, count in block["proposal"]["rejected"].items():
            scores = [r["cosine"] for r in block["abstention"] if r["group"] == group]
            assert count == len(scores) - ab.share_at_least(scores, threshold)[0]
        assert block["table"] == ab.threshold_table(pos, {g: [r["cosine"] for r in block["abstention"] if r["group"] == g] for g in ab.GROUPS})
        assert all(-1.0 <= x <= 1.0 for x in pos + neg)  # un coseno


# ---------- con Postgres ----------


def test_top_chunks_reproduce_the_saved_cosines(saved):
    _, vectors = embeddings.load_embeddings("abstention_queries")
    try:
        found = ab.top_chunks("window", vectors[:3])
    except Exception as error:  # sin Postgres o sin corpus cargado (en la CI es un fallo, no un salto)
        skip_or_fail(f"base de datos no disponible o sin cargar: {error}")
    stored = saved["chunkings"]["window"]["abstention"][:3]
    assert [f["chunk_id"] for f in found] == [s["chunk_id"] for s in stored]
    assert np.allclose([f["cosine"] for f in found], [s["cosine"] for s in stored], atol=1e-9)


# ---------- el umbral que lee la API ----------


def test_floor_to_thousandths_rounds_down_and_keeps_exact_values():
    assert ab.floor_to_thousandths(0.60977) == 0.609 and ab.floor_to_thousandths(0.5899) == 0.589
    assert ab.floor_to_thousandths(0.61) == 0.61 and ab.floor_to_thousandths(0.589) == 0.589  # ya en milésimas: no baja otra

def test_load_thresholds(tmp_path):
    path = tmp_path / "threshold.json"
    assert ab.load_thresholds(path) == {}  # sin fichero, sin abstención
    path.write_text(json.dumps({"thresholds": {"window": 0.6}}))
    assert ab.load_thresholds(path) == {"window": 0.6}
    for bad in ({}, {"window": 1.5}, {"window": "x"}, {"window": 0.6, "article": 0.5}):  # article: la API no se abstiene con artículos
        path.write_text(json.dumps({"thresholds": bad}))
        with pytest.raises(RuntimeError, match="umbrales"):
            ab.load_thresholds(path)


def test_the_versioned_threshold_file_follows_from_the_saved_result(saved):
    spec = json.loads(ab.THRESHOLD_FILE.read_text(encoding="utf-8"))
    assert set(spec["thresholds"]) == {"window"}  # la API solo se abstiene con ventanas
    assert spec["thresholds"] == {c: ab.floor_to_thousandths(saved["chunkings"][c]["proposal"]["threshold"]) for c in ab.ABSTENTION_CHUNKINGS}
    assert spec["calibrated_on"] == saved["inputs"], "threshold.json está desfasado: ejecuta `python cli.py abstention --write-threshold`"
    assert ab.load_thresholds() == spec["thresholds"] and spec["retention"] == ab.RETENTION
