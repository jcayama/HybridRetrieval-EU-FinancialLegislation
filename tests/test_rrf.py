import pytest

from scripts.rrf import K_DEFAULT, rrf, rrf_scores

# Ejemplo hecho a mano (k = 60): A = [x, y, z], B = [y, z, w]
#   y = 1/62 + 1/61 = 0.032522   z = 1/63 + 1/62 = 0.032002   x = 1/61 = 0.016393   w = 1/63 = 0.015873
A = ["x", "y", "z"]
B = ["y", "z", "w"]


def test_default_k_is_60():
    assert K_DEFAULT == 60


def test_scores_hand_example():
    scores = rrf_scores([A, B])
    assert scores["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["z"] == pytest.approx(1 / 63 + 1 / 62)
    assert scores["x"] == pytest.approx(1 / 61)
    assert scores["w"] == pytest.approx(1 / 63)


def test_order_hand_example():
    assert rrf([A, B]) == ["y", "z", "x", "w"]


def test_single_ranking_keeps_its_order():
    assert rrf([["c", "a", "b"]]) == ["c", "a", "b"]


def test_items_in_only_one_ranking_are_included():
    assert set(rrf([["a"], ["b"]])) == {"a", "b"}


def test_ties_are_broken_by_id():
    # [a, b] y [b, a]: puntuaciones idénticas; el desempate por id da un orden determinista
    assert rrf([["a", "b"], ["b", "a"]]) == ["a", "b"]
    assert rrf([["b", "a"], ["a", "b"]]) == ["a", "b"]


def test_k_changes_the_balance():
    # R1 = [p, f, q] y R2 = [g, h, q]. q es 3.º en los dos rankings; p y g son 1.º en uno solo.
    #   k = 0:  p = g = 1, q = 1/3 + 1/3 = 0.667  -> gana p o g (ser primero pesa mucho)
    #   k = 60: p = g = 1/61 = 0.0164, q = 2/63 = 0.0317  -> gana q (aparecer en los dos rankings)
    rankings = [["p", "f", "q"], ["g", "h", "q"]]
    assert rrf(rankings, k=0)[0] in {"p", "g"}
    assert rrf(rankings, k=60)[0] == "q"


def test_empty_inputs():
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_k_must_not_be_negative():
    with pytest.raises(ValueError, match="k"):
        rrf([A], k=-1)


def test_duplicates_inside_one_ranking_are_rejected():
    with pytest.raises(ValueError, match="repetid"):
        rrf([["a", "a", "b"]])


def test_order_of_the_rankings_does_not_matter():
    assert rrf([A, B]) == rrf([B, A])
