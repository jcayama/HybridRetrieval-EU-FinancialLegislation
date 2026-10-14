from scripts.hybrid import RrfIndex


class FakeIndex:
    """Índice falso que devuelve siempre el mismo ranking de chunks."""

    def __init__(self, chunks, info=None):
        self.chunks = chunks
        self.info = info or {}

    def search_chunks(self, query, depth):
        return self.chunks[:depth]


def entry(chunk_id, article):
    return chunk_id, ("C", article)


# léxico: [c1, c2]   denso: [c2, c3, c1]   ->   RRF (k=60): c2 = 1/62 + 1/61, c1 = 1/61 + 1/63, c3 = 1/62
LEXICAL = FakeIndex([entry("c1", 1), entry("c2", 2)], {"lexical_marker": True})
DENSE = FakeIndex([entry("c2", 2), entry("c3", 3), entry("c1", 1)], {"dense_marker": True})


def test_rrf_index_fuses_lexical_and_dense_rankings():
    index = RrfIndex(LEXICAL, DENSE, "bm25")
    assert index.search_chunks("q", 10) == [entry("c2", 2), entry("c1", 1), entry("c3", 3)]
    assert index.search("q", 10) == [("C", 2), ("C", 1), ("C", 3)]


def test_rrf_index_respects_depth():
    assert len(RrfIndex(LEXICAL, DENSE, "bm25").search_chunks("q", 2)) == 2


def test_rrf_index_documents_its_inputs_and_merges_info():
    info = RrfIndex(LEXICAL, DENSE, "bm25", {"exploratory": True}).info
    assert info["inputs"] == ["bm25", "dense"] and info["fusion"] == "RRF" and info["k"] == 60
    assert info["exploratory"] is True and info["lexical_marker"] and info["dense_marker"]


def test_rrf_index_with_one_empty_side_equals_the_other():
    empty = FakeIndex([])
    assert RrfIndex(empty, DENSE, "fts").search_chunks("q", 10) == DENSE.chunks
    assert RrfIndex(LEXICAL, empty, "fts").search_chunks("q", 10) == LEXICAL.chunks


def test_two_chunks_of_the_same_article_stay_separate():
    lexical = FakeIndex([entry("w0", 7), entry("w1", 7)])
    fused = RrfIndex(lexical, FakeIndex([entry("w1", 7)]), "bm25").search_chunks("q", 10)
    assert [chunk_id for chunk_id, _ in fused] == ["w1", "w0"]
