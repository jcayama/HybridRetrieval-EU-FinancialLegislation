import pytest

from scripts.browse_corpus import ALIASES, get_article, list_articles, parse_range, resolve_act
from scripts.chunk_corpus import ARTICLES_FILE


def test_resolve_act_accepts_alias_and_celex():
    assert resolve_act("DORA") == "32022R2554"
    assert resolve_act("32022R2554") == "32022R2554"
    with pytest.raises(ValueError):
        resolve_act("mifid")


def test_parse_range():
    assert parse_range("17-23") == (17, 23)
    assert parse_range("5") == (5, 5)
    with pytest.raises(ValueError):
        parse_range("9-3")


chunks = pytest.mark.skipif(not ARTICLES_FILE.exists(), reason="chunks no generados (python cli.py chunk)")


@chunks
def test_list_articles_filters():
    rows = list_articles("dora", "17-23")
    assert len(rows) == 7
    assert all(r.startswith("dora") for r in rows)
    assert len(list_articles()) == 248


@chunks
def test_get_article():
    article = get_article("dora", 26)
    assert article["celex"] == ALIASES["dora"] and article["article"] == 26
    assert article["text"].startswith("Article 26.")
    with pytest.raises(KeyError):
        get_article("dora", 999)
