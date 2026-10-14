import pytest

from scripts import db
from scripts.fts import FtsIndex, build_or_query

# ---------- sin base de datos ----------


def test_build_or_query_joins_distinct_terms_with_or():
    assert build_or_query("How quickly, must a company tell regulators?") == "how | quickly | must | a | company | tell | regulators"
    assert build_or_query("depositary depositary Depositary") == "depositary"


def test_build_or_query_cannot_inject_tsquery_operators():
    assert build_or_query("a & !b | (c) <-> d:*") == "a | b | c | d"
    assert build_or_query("   ?!  ") is None
    assert build_or_query("") is None


def test_conninfo_requires_credentials(monkeypatch):
    monkeypatch.setattr(db, "load_dotenv", lambda *a, **k: None)
    for var in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB", "POSTGRES_HOST", "POSTGRES_PORT"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match="POSTGRES_USER"):
        db.conninfo()


def test_conninfo_defaults_to_loopback_and_5433(monkeypatch):
    monkeypatch.setattr(db, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("POSTGRES_USER", "u")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p")
    monkeypatch.setenv("POSTGRES_DB", "d")
    monkeypatch.setenv("POSTGRES_HOST", "")  # vacío = no definido
    monkeypatch.setenv("POSTGRES_PORT", "")
    info = db.conninfo()
    assert info["host"] == "127.0.0.1" and info["port"] == 5433 and info["dbname"] == "d"


# ---------- con base de datos (se saltan si no hay Postgres) ----------


def chunk(strategy, celex, article, text, **extra):
    base = {"chunk_id": f"{celex}:art_{article}:{strategy}", "strategy": strategy, "celex": celex, "act": "A", "article": article, "title": "T", "chapter_id": None, "text": text, "n_words": len(text.split())}
    return base | extra


def load(schema, strategy, chunks, sha="sha"):
    with db.connect(schema) as conn:
        return db.load_chunks(conn, strategy, chunks, sha)


def test_init_db_is_idempotent(schema):
    with db.connect(schema) as conn:
        db.init_db(conn)
        db.init_db(conn)


def test_load_replaces_previous_chunks_and_records_hash(schema):
    chunks = [chunk("article", "C", 1, "alpha"), chunk("article", "C", 2, "beta")]
    assert load(schema, "article", chunks, "h1") == 2
    assert load(schema, "article", chunks[:1], "h2") == 1
    with db.connect(schema) as conn:
        assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1
        assert conn.execute("SELECT chunks_sha256, n_chunks FROM load_info WHERE strategy = 'article'").fetchone() == ("h2", 1)


def test_search_ranks_matches_and_applies_stemming(schema):
    chunks = [
        chunk("article", "C", 1, "remuneration policies of the manager"),
        chunk("article", "C", 2, "unrelated marketing rules"),
        chunk("article", "C", 3, "remuneration remuneration remuneration policies"),
    ]
    load(schema, "article", chunks, "s")
    index = FtsIndex("article", "s", schema)
    assert index.search("remunerations policy", 10) == [("C", 3), ("C", 1)]  # 'english': remunerations ~ remuneration, policy ~ policies


def test_search_uses_or_semantics(schema):
    load(schema, "article", [chunk("article", "C", 1, "depositary safekeeping of assets")], "s")
    index = FtsIndex("article", "s", schema)
    assert index.search("depositary zzzunknownzzz", 10) == [("C", 1)]  # con AND no devolvería nada


def test_search_filters_by_strategy_and_respects_depth(schema):
    load(schema, "article", [chunk("article", "C", 1, "depositary assets")], "a")
    load(schema, "window", [chunk("window", "C", 7, "depositary assets one", window=0), chunk("window", "C", 7, "depositary assets two", chunk_id="C:art_7:w200:1", window=1)], "w")
    assert FtsIndex("article", "a", schema).search("depositary", 10) == [("C", 1)]
    windows = FtsIndex("window", "w", schema)
    assert windows.search("depositary", 10) == [("C", 7), ("C", 7)]  # un artículo puede repetirse (dos ventanas)
    assert len(windows.search("depositary", 1)) == 1


def test_search_ties_are_broken_by_chunk_id(schema):
    # 40 chunks con el mismo texto (misma puntuación) e ids desordenados: sin desempate explícito no hay orden garantizado
    numbers = [(i * 17) % 40 for i in range(40)]
    chunks = [chunk("article", "C", n + 1, "same text here", chunk_id=f"id{n:02d}") for n in numbers]
    load(schema, "article", chunks, "s")
    assert FtsIndex("article", "s", schema).search("same text", 10) == [("C", n + 1) for n in range(10)]


def test_search_with_no_usable_terms_returns_nothing(schema):
    load(schema, "article", [chunk("article", "C", 1, "something")], "s")
    index = FtsIndex("article", "s", schema)
    assert index.search("??", 10) == []
    assert index.search("the of and", 10) == []  # solo palabras vacías: tsquery vacío


def test_stale_or_missing_database_is_detected(schema):
    with pytest.raises(RuntimeError, match="load-db"):
        FtsIndex("article", "s", schema)  # nada cargado
    load(schema, "article", [chunk("article", "C", 1, "x")], "old")
    with pytest.raises(RuntimeError, match="no coinciden"):
        FtsIndex("article", "new", schema)
