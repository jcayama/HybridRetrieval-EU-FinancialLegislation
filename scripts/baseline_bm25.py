"""Baseline léxico BM25 (Okapi) en Python, con `rank-bm25` como método de referencia.

Tokenización v0, deliberadamente simple: minúsculas y tokens alfanuméricos; sin stemming ni stopwords. Los
parámetros de BM25 son los valores por defecto de la librería (k1=1.5, b=0.75, epsilon=0.25) y NO se ajustan
con el golden set. El FTS de Postgres (configuración 'english') llegará después como método propio.
"""

import re

import numpy as np
from rank_bm25 import BM25Okapi

from scripts.metrics import Article

K1, B, EPSILON = 1.5, 0.75, 0.25


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Bm25Index:
    def __init__(self, chunks: list[dict]):
        if not chunks:
            raise ValueError("no hay chunks que indexar")
        self.chunk_ids: list[str] = [c["chunk_id"] for c in chunks]
        self.articles: list[Article] = [(c["celex"], c["article"]) for c in chunks]
        self.model = BM25Okapi([tokenize(c["text"]) for c in chunks], k1=K1, b=B, epsilon=EPSILON)

    def search_chunks(self, query: str, depth: int) -> list[tuple[str, Article]]:
        """Los `depth` mejores chunks como (chunk_id, artículo). Solo puntuaciones > 0."""
        scores = self.model.get_scores(tokenize(query))
        order = np.argsort(-scores, kind="stable")[:depth]  # estable: a igual puntuación, orden del corpus
        return [(self.chunk_ids[i], self.articles[i]) for i in order if scores[i] > 0]

    def search(self, query: str, depth: int) -> list[Article]:
        """Los `depth` mejores chunks, como artículos (un artículo puede repetirse)."""
        return [article for _, article in self.search_chunks(query, depth)]
