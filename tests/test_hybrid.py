import numpy as np

from scripts import db, embeddings
from scripts.embeddings import DIM
from scripts.hybrid import HybridIndex


def unit(*pairs):
    v = np.zeros(DIM, dtype=np.float32)
    for position, value in pairs:
        v[position] = value
    return v / np.linalg.norm(v)


def chunk(strategy, article, text, chunk_id=None):
    return {
        "chunk_id": chunk_id or f"C:art_{article}:{strategy}",
        "strategy": strategy,
        "celex": "C",
        "act": "A",
        "article": article,
        "title": "T",
        "chapter_id": None,
        "text": text,
        "n_words": len(text.split()),
    }


def load(schema, strategy, rows, sha="c"):
    """rows: lista de (chunk, vector)."""
    emb_sha = embeddings.read_manifest()["files"][f"chunks_{strategy}"]["sha256"]
    with db.connect(schema) as conn:
        db.load_chunks(conn, strategy, [c for c, _ in rows], sha, {c["chunk_id"]: v for c, v in rows}, emb_sha)


def hybrid(schema, strategy, query_vector):
    return HybridIndex(strategy, "c", lambda query: query_vector, schema)


# FTS para «alpha beta»: c1 (los dos términos) > c2 (uno)  | denso con q = unit((1,1),(2,0.5)): c2 > c3 > c1
# RRF (k=60): c1 = 1/61 + 1/63 = 0.03227, c2 = 1/62 + 1/61 = 0.03252, c3 = 1/62 = 0.01613  ->  c2, c1, c3
ROWS = [
    (chunk("article", 1, "alpha beta"), unit((0, 1))),
    (chunk("article", 2, "alpha"), unit((1, 1))),
    (chunk("article", 3, "gamma"), unit((2, 1))),
]
QUERY_VECTOR = unit((1, 1), (2, 0.5))


def test_each_system_alone_disagrees_on_the_order(schema):
    load(schema, "article", ROWS)
    index = hybrid(schema, "article", QUERY_VECTOR)
    assert index.fts.search("alpha beta", 10) == [("C", 1), ("C", 2)]
    assert index.dense.search("alpha beta", 10) == [("C", 2), ("C", 3), ("C", 1)]


def test_hybrid_fuses_with_rrf_hand_example(schema):
    load(schema, "article", ROWS)
    assert hybrid(schema, "article", QUERY_VECTOR).search("alpha beta", 10) == [("C", 2), ("C", 1), ("C", 3)]


def test_hybrid_falls_back_to_dense_when_fts_finds_nothing(schema):
    load(schema, "article", ROWS)
    index = hybrid(schema, "article", QUERY_VECTOR)
    assert index.search("zzzunknownzzz", 10) == index.dense.search("zzzunknownzzz", 10)


def test_fusion_is_at_chunk_level_so_windows_of_one_article_stay_separate(schema):
    rows = [
        (chunk("window", 7, "alpha one", chunk_id="w0"), unit((0, 1))),
        (chunk("window", 7, "alpha two", chunk_id="w1"), unit((0, 1), (1, 0.1))),
        (chunk("window", 8, "alpha three", chunk_id="w2"), unit((1, 1))),
    ]
    load(schema, "window", rows)
    chunks = hybrid(schema, "window", unit((0, 1))).search_chunks("alpha", 10)
    assert {chunk_id for chunk_id, _ in chunks} == {"w0", "w1", "w2"}  # tres chunks, aunque dos sean del mismo artículo
    assert [article for _, article in chunks].count(("C", 7)) == 2


def test_depth_truncates_the_fused_ranking(schema):
    load(schema, "article", ROWS)
    assert len(hybrid(schema, "article", QUERY_VECTOR).search("alpha beta", 2)) <= 2


def test_info_documents_the_fusion(schema):
    load(schema, "article", ROWS)
    info = hybrid(schema, "article", QUERY_VECTOR).info
    assert info["fusion"] == "RRF" and info["k"] == 60 and info["inputs"] == ["fts", "dense"]
    assert info["model"] == embeddings.MODEL_ID and "postgres" in info and "pgvector" in info
