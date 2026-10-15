"""El servicio de recuperación contra Postgres real (esquema temporal aislado) y con inquilinos."""

import numpy as np
import psycopg
import pytest

from functions.retriever import DatabaseNotReady, ModelUnavailable, RetrievalService
from scripts import db, embeddings
from scripts import evaluate as ev
from scripts.download_corpus import sha256_bytes
from scripts.embeddings import DIM


def unit(*pairs):
    v = np.zeros(DIM, dtype=np.float32)
    for position, value in pairs:
        v[position] = value
    return v / np.linalg.norm(v)


def row(chunk_id, strategy, article, text, vector, tenant=None):
    return {
        "chunk_id": chunk_id,
        "strategy": strategy,
        "celex": "C1",
        "act": "Acto de prueba",
        "article": article,
        "title": f"Título {article}",
        "text": text,
        "n_words": len(text.split()),
        "embedding": db.to_vector_literal(vector),
        "tenant": tenant,
    }


# Corpus mínimo, con el formato real de los chunks («Article N. Título» + cuerpo)
ROWS = [
    row("w7a", "window", 7, "Article 7. Título 7\nalpha beta gamma first window", unit((0, 1))),
    row("w7b", "window", 7, "Article 7. Título 7\nalpha window two delta", unit((0, 1), (1, 0.2))),
    row("w8", "window", 8, "Article 8. Título 8\nbeta epsilon", unit((1, 1))),
    row("a7", "article", 7, "Article 7. Título 7\nalpha beta gamma delta", unit((0, 1))),
    row("a8", "article", 8, "Article 8. Título 8\nbeta epsilon", unit((1, 1))),
]
TENANT_ROWS = [
    row("wA", "window", 101, "Article 101. Título 101\nalpha CANARYA private policy of tenant A", unit((2, 1)), tenant="A"),
    row("wB", "window", 201, "Article 201. Título 201\nalpha CANARYB private policy of tenant B", unit((3, 1)), tenant="B"),
]


def seed(schema, rows=ROWS):
    sql = (
        "INSERT INTO chunks (chunk_id, strategy, celex, act, article, title, text, n_words, embedding, tenant_id) "
        "VALUES (%(chunk_id)s, %(strategy)s, %(celex)s, %(act)s, %(article)s, %(title)s, %(text)s, %(n_words)s, %(embedding)s::vector, %(tenant)s)"
    )
    manifest = embeddings.read_manifest()["files"]
    with db.connect(schema) as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
        for strategy in ("article", "window"):
            sha = sha256_bytes(ev.CHUNK_FILES[strategy].read_bytes())
            conn.execute("INSERT INTO load_info (strategy, chunks_sha256, n_chunks, embeddings_sha256) VALUES (%s, %s, 0, %s)", (strategy, sha, manifest[f"chunks_{strategy}"]["sha256"]))


def service(schema, embedder=True, tenants=False):
    return RetrievalService(schema=schema, embed_query=(lambda q: unit((0, 1))) if embedder else None, enable_tenants=tenants)


# ---------- FTS: no necesita modelo ----------


def test_fts_returns_distinct_articles_with_title_and_snippet(schema):
    seed(schema)
    hits = service(schema, embedder=False).search("alpha beta", top_k=10, mode="fts", chunking="window")
    assert [(h.celex, h.article) for h in hits] == [("C1", 7), ("C1", 8)]  # w7a y w7b son del artículo 7: aparece una sola vez
    assert [h.rank for h in hits] == [1, 2] and hits[0].title == "Título 7" and hits[0].act == "Acto de prueba"
    assert hits[0].snippet == "alpha beta gamma first window"  # sin la cabecera «Article N. Título»


def test_the_snippet_comes_from_the_best_matching_chunk_of_each_article(schema):
    seed(schema)
    hit = service(schema, embedder=False).search("window two delta", mode="fts", chunking="window")[0]
    assert hit.article == 7 and hit.chunk_id == "w7b" and "two delta" in hit.snippet


def test_top_k_limits_the_number_of_articles(schema):
    seed(schema)
    assert len(service(schema, embedder=False).search("beta", top_k=1, mode="fts", chunking="window")) == 1
    assert len(service(schema, embedder=False).search("beta", top_k=10, mode="fts", chunking="article")) == 2


def test_long_chunks_are_cut_into_a_short_snippet(schema):
    long_text = "Article 9. Título 9\n" + "palabra " * 200
    seed(schema, ROWS + [row("w9", "window", 9, long_text, unit((4, 1)))])
    snippet = service(schema, embedder=False).search("palabra", mode="fts", chunking="window")[0].snippet
    assert len(snippet) <= 410 and snippet.endswith("…") and not snippet.startswith("Article")


def test_a_query_without_matches_returns_an_empty_list(schema):
    seed(schema)
    assert service(schema, embedder=False).search("zzzunknownzzz", mode="fts") == []
    assert service(schema, embedder=False).search("???", mode="fts") == []


# ---------- denso e híbrido: necesitan el modelo ----------


def test_dense_and_hybrid_need_the_model(schema):
    seed(schema)
    svc = service(schema, embedder=False)
    for mode in ("dense", "hybrid"):
        with pytest.raises(ModelUnavailable):
            svc.search("alpha", mode=mode)
    assert svc.search("alpha", mode="fts")  # FTS sigue funcionando sin modelo


def test_dense_orders_by_cosine_and_hybrid_fuses_both_rankings(schema):
    seed(schema)
    svc = service(schema)  # el embebedor falso devuelve el vector del artículo 7
    assert [h.article for h in svc.search("cualquier cosa", mode="dense", chunking="article")] == [7, 8]
    assert {h.article for h in svc.search("alpha beta", mode="hybrid", chunking="window")} == {7, 8}


# ---------- inquilinos ----------


def test_without_tenants_enabled_only_the_public_corpus_is_visible_even_if_a_tenant_is_passed(schema):
    seed(schema, ROWS + TENANT_ROWS)
    hits = service(schema, embedder=False, tenants=False).search("alpha", mode="fts", chunking="window", tenant="A")
    assert {h.article for h in hits} == {7}  # nada de 101 ni 201


def test_a_tenant_sees_the_public_corpus_and_its_own_documents_but_never_the_others(schema):
    seed(schema, ROWS + TENANT_ROWS)
    svc = service(schema, embedder=False, tenants=True)
    as_a = {h.article for h in svc.search("alpha", mode="fts", chunking="window", tenant="A")}
    as_b = {h.article for h in svc.search("alpha", mode="fts", chunking="window", tenant="B")}
    assert as_a == {7, 101} and as_b == {7, 201}
    assert svc.search("CANARYA", mode="fts", chunking="window", tenant="B") == []  # el testigo de A es invisible para B
    assert {h.article for h in svc.search("alpha", mode="fts", chunking="window", tenant=None)} == {7}  # sin cabecera: solo lo público


def test_dense_and_hybrid_also_respect_tenants(schema):
    seed(schema, ROWS + TENANT_ROWS)
    svc = RetrievalService(schema=schema, embed_query=lambda q: unit((2, 1)), enable_tenants=True)  # la consulta apunta al vector privado de A
    for mode in ("dense", "hybrid"):
        assert 101 not in {h.article for h in svc.search("alpha", mode=mode, chunking="window", tenant="B")}
        assert 101 in {h.article for h in svc.search("alpha", mode=mode, chunking="window", tenant="A")}


# ---------- estado y fallos de la base de datos ----------


def test_status_reports_database_model_and_tenants(schema):
    seed(schema)
    assert service(schema, embedder=False).status() == {"database": True, "model": False, "tenants_enabled": False}
    assert service(schema, tenants=True).status() == {"database": True, "model": True, "tenants_enabled": True}


def test_an_empty_database_is_reported_as_not_ready(schema):
    with pytest.raises(DatabaseNotReady, match="load-db"):
        service(schema, embedder=False).search("alpha", mode="fts")


def test_an_unreachable_database_is_reported_as_not_ready_and_status_is_false(schema, monkeypatch):
    def refuse(*args, **kwargs):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(db, "connect", refuse)
    svc = service(schema, embedder=False)
    assert svc.status()["database"] is False
    with pytest.raises(DatabaseNotReady):
        svc.search("alpha", mode="fts")
