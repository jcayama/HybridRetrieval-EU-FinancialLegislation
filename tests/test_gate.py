import pytest

from scripts import gate


def test_round_helpers_use_multiples_of_0_005():
    assert gate.round_up(0.024) == 0.025 and gate.round_up(0.025) == 0.025 and gate.round_up(0.0251) == 0.03
    assert gate.round_down(0.802) == 0.800 and gate.round_down(0.8) == 0.8 and gate.round_down(0.8049) == 0.8
    assert gate.round_down(0.806) == 0.805


@pytest.mark.parametrize(
    ("noise", "margin"),
    [(0.0, 0.01), (0.002, 0.01), (0.0034, 0.015), (0.008, 0.025), (0.01, 0.03)],
)
def test_margin_is_three_times_the_noise_with_a_minimum_of_0_01(noise, margin):
    assert gate.margin_for(noise) == pytest.approx(margin)


def test_floor_is_the_current_value_minus_the_margin_rounded_down():
    assert gate.floor_for(0.827, 0.025) == 0.8
    assert gate.floor_for(0.667, 0.025) == 0.64
    assert gate.floor_for(0.641, 0.01) == 0.63


def test_check_passes_at_or_above_the_floor_and_fails_below():
    thresholds = {"article": {"recall@10": 0.80, "ndcg@10": 0.60}, "window": {"recall@10": 0.85}}
    assert gate.check({"article": {"recall@10": 0.80, "ndcg@10": 0.61}, "window": {"recall@10": 0.9}}, thresholds) == []  # el umbral exacto pasa
    failures = gate.check({"article": {"recall@10": 0.799, "ndcg@10": 0.7}, "window": {"recall@10": 0.9}}, thresholds)
    assert len(failures) == 1 and "article recall@10" in failures[0]


def test_check_reports_missing_values_instead_of_passing():
    failures = gate.check({"article": {"recall@10": 0.9}}, {"article": {"recall@10": 0.8, "ndcg@10": 0.6}, "window": {"recall@10": 0.8}})
    assert any("article ndcg@10: sin valor" in f for f in failures) and any("window recall@10: sin valor" in f for f in failures)


def test_committed_thresholds_are_consistent_with_the_rule():
    import json

    if not gate.THRESHOLDS_FILE.exists():
        pytest.skip("umbrales no calculados (python cli.py calibrate-gate --write-thresholds)")
    spec = json.loads(gate.THRESHOLDS_FILE.read_text())
    assert spec["margin"] >= gate.MIN_MARGIN
    for floors in spec["thresholds"].values():
        assert set(floors) == set(gate.METRICS) and all(0 < v < 1 and round(v / gate.ROUND, 6) == round(v / gate.ROUND) for v in floors.values())


def test_thresholds_for_applies_the_margin_to_every_gated_metric():
    current = {c: {"recall@10": 0.827, "ndcg@10": 0.641} for c in gate.CHUNKINGS}
    th = gate.thresholds_for(current, 0.01)
    assert th["article"] == {"recall@10": 0.815, "ndcg@10": 0.63}
    assert gate.thresholds_for(current, 0.025)["window"] == {"recall@10": 0.8, "ndcg@10": 0.615}
    assert gate.check(current, th) == []  # el valor actual siempre supera su propio umbral


def test_margin_below_the_minimum_is_rejected():
    assert gate.run_set_margin(0.005) == 1


def _spec(tmp_path, floor):
    import json

    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"system": "x", "margin": 0.01, "thresholds": {c: {"recall@10": floor, "ndcg@10": floor} for c in gate.CHUNKINGS}}))
    return path


def test_run_gate_exits_0_when_every_metric_is_above_its_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "THRESHOLDS_FILE", _spec(tmp_path, 0.5))
    monkeypatch.setattr(gate, "current_values", lambda: {c: {"recall@10": 0.9, "ndcg@10": 0.8} for c in gate.CHUNKINGS})
    assert gate.run_gate() == 0


def test_run_gate_exits_1_when_a_metric_drops_below_its_floor(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gate, "THRESHOLDS_FILE", _spec(tmp_path, 0.5))
    monkeypatch.setattr(gate, "current_values", lambda: {"article": {"recall@10": 0.9, "ndcg@10": 0.8}, "window": {"recall@10": 0.49, "ndcg@10": 0.8}})
    assert gate.run_gate() == 1
    assert "window recall@10" in capsys.readouterr().err


def test_run_gate_exits_1_when_thresholds_are_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "THRESHOLDS_FILE", tmp_path / "no_existe.json")
    assert gate.run_gate() == 1
