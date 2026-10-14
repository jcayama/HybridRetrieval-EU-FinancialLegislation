"""Reciprocal Rank Fusion (RRF), como función pura.

Cada elemento suma 1 / (k + posición) por cada ranking en el que aparece (posición empezando en 1). Con k = 60,
el valor habitual de la literatura, que suaviza la diferencia entre las primeras posiciones. Desempate por id,
para que el resultado sea determinista.
"""

K_DEFAULT = 60


def rrf_scores(rankings: list[list[str]], k: int = K_DEFAULT) -> dict[str, float]:
    if k < 0:
        raise ValueError(f"k no puede ser negativo (recibido {k})")
    scores: dict[str, float] = {}
    for ranking in rankings:
        if len(set(ranking)) != len(ranking):
            raise ValueError("un ranking contiene elementos repetidos")
        for position, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position)
    return scores


def rrf(rankings: list[list[str]], k: int = K_DEFAULT) -> list[str]:
    """Ids fusionados, de mayor a menor puntuación RRF (desempate por id)."""
    scores = rrf_scores(rankings, k)
    return sorted(scores, key=lambda item: (-scores[item], item))
