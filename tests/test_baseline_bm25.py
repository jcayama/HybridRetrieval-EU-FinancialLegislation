import pytest

from scripts.baseline_bm25 import Bm25Index, tokenize

_counter = iter(range(10_000))


def chunk(celex, article, text):
    return {"chunk_id": f"{celex}:art_{article}:{next(_counter)}", "celex": celex, "article": article, "text": text}


CHUNKS = [
    chunk("C", 1, "Article 1. Alpha\nThe depositary shall safe-keep the assets of the fund."),
    chunk("C", 2, "Article 2. Beta\nRemuneration policies shall be consistent with sound risk management."),
    chunk("C", 3, "Article 3. Gamma\nThe competent authority may withdraw the authorisation."),
    chunk("C", 3, "Article 3. Gamma\nA second window of the same article about the depositary."),
    *[chunk("C", 10 + i, f"Article {10 + i}. Filler\nUnrelated provisions about marketing and reporting number {i}.") for i in range(6)],
]  # los chunks de relleno evitan IDF nulo: un término presente en la mitad de los chunks puntúa 0 en BM25 Okapi


def test_tokenize():
    assert tokenize("Safe-keeping, EUR 125 000!") == ["safe", "keeping", "eur", "125", "000"]


def test_search_ranks_matching_chunk_first():
    index = Bm25Index(CHUNKS)
    assert index.search("remuneration policies", 10)[0] == ("C", 2)
    assert index.search("withdraw authorisation", 10)[0] == ("C", 3)


def test_search_returns_one_entry_per_chunk_so_articles_can_repeat():
    index = Bm25Index(CHUNKS)
    assert index.search("depositary", 10).count(("C", 3)) == 1
    assert set(index.search("depositary", 10)) == {("C", 1), ("C", 3)}


def test_search_drops_zero_score_chunks_and_respects_depth():
    index = Bm25Index(CHUNKS)
    assert index.search("xyzzy", 10) == []  # ningún término coincide
    assert len(index.search("the", 2)) <= 2


def test_search_is_deterministic():
    index = Bm25Index(CHUNKS)
    assert index.search("depositary assets", 10) == index.search("depositary assets", 10)


def test_empty_corpus_is_rejected():
    with pytest.raises(ValueError):
        Bm25Index([])


def test_search_chunks_returns_chunk_ids_matching_the_articles():
    index = Bm25Index(CHUNKS)
    results = index.search_chunks("depositary", 10)
    assert {article for _, article in results} == {("C", 1), ("C", 3)}
    assert len({chunk_id for chunk_id, _ in results}) == len(results)  # ids únicos
    assert [article for _, article in results] == index.search("depositary", 10)
