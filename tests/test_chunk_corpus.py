import json

import pytest
from bs4 import BeautifulSoup

from scripts.chunk_corpus import (
    ARTICLES_FILE,
    MIN_TAIL,
    OVERLAP,
    WINDOW,
    WINDOWS_FILE,
    article_divs,
    build_all,
    build_chunks,
    chars,
    dump_jsonl,
    extract_article,
    header,
    window_bounds,
)
from scripts.download_corpus import ACTS, HASH_FILE, RAW_DIR

XML = """<html><body>
<div class="cpt"><div id="cpt_II.sct_I">
<div class="eli-subdivision" id="art_5">
  <p class="oj-ti-art">Article 5</p>
  <div class="eli-title" id="art_5.tit_1"><p class="oj-sti-art">Governance</p></div>
  <div id="005.001"><p class="oj-normal">1.   Entities shall:</p>
    <table><tbody><tr><td><p>(a)</p></td><td><p>bear the responsibility;</p></td></tr></tbody></table>
    <table><tbody><tr><td><p>(b)</p></td><td><p>set clear roles.</p></td></tr></tbody></table>
  </div>
  <div id="005.002"><p class="oj-normal">2.   Second paragraph.</p></div>
</div></div></div></body></html>"""


def test_table_rows_become_single_lines_and_header_is_excluded():
    art = extract_article(article_divs(BeautifulSoup(XML, "xml"))[0])
    assert art["number"] == 5
    assert art["title"] == "Governance"
    assert art["chapter_id"] == "cpt_II.sct_I"
    assert art["lines"] == ["1. Entities shall:", "(a) bear the responsibility;", "(b) set clear roles.", "2. Second paragraph."]


def test_header():
    assert header(5, "Governance") == "Article 5. Governance"
    assert header(5, "") == "Article 5"


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, [(0, 0)]),
        (100, [(0, 100)]),
        (200, [(0, 200)]),
        (201, [(0, 201)]),  # cola de 1 palabra: se fusiona
        (249, [(0, 249)]),  # cola de 49 (< 50): se fusiona
        (250, [(0, 200), (160, 250)]),  # cola de 50: ventana propia, con 40 de solape
        (360, [(0, 200), (160, 360)]),
        (400, [(0, 200), (160, 400)]),  # cola de 40 (< 50): se fusiona con la anterior
        (450, [(0, 200), (160, 360), (320, 450)]),  # cola de 90: ventana propia
    ],
)
def test_window_bounds_cases(n, expected):
    assert window_bounds(n) == expected


@pytest.mark.parametrize("n", range(1, 1500))
def test_window_bounds_invariants(n):
    bounds = window_bounds(n)
    assert bounds[0][0] == 0 and bounds[-1][1] == n  # cubren todo el texto
    for (s1, e1), (s2, _) in zip(bounds, bounds[1:], strict=False):
        assert s2 == s1 + (WINDOW - OVERLAP)  # paso fijo
        assert s2 < e1  # siempre hay solape: no se pierde ninguna palabra
    assert all(e - s <= WINDOW + MIN_TAIL - 1 for s, e in bounds)  # máximo 249 con la fusión
    assert all(e - s >= 1 for s, e in bounds)


def test_build_chunks_synthetic():
    act = ACTS[1]
    articles, windows = build_chunks(act, XML.encode())
    assert len(articles) == 1 and len(windows) == 1
    assert articles[0]["chunk_id"] == "32022R2554:art_5"
    assert articles[0]["text"].startswith("Article 5. Governance\n1. Entities shall:\n(a) bear")
    assert windows[0]["chunk_id"] == "32022R2554:art_5:w200:0"
    assert windows[0]["text"].startswith("Article 5. Governance\n1. Entities shall:")


real_data = pytest.mark.skipif(
    not all((RAW_DIR / a.filename).exists() for a in ACTS) or not HASH_FILE.exists(),
    reason="corpus no descargado (python cli.py download-corpus)",
)


@real_data
def test_real_corpus_counts_and_fidelity():
    articles, windows = build_all()
    assert len(articles) == sum(a.expected_articles for a in ACTS)  # 248

    # Fidelidad: cada artículo conserva TODOS sus caracteres (salvo espacios) respecto al HTML original.
    for act in ACTS:
        soup = BeautifulSoup((RAW_DIR / act.filename).read_bytes(), "xml")
        by_number = {a["article"]: a for a in articles if a["celex"] == act.celex}
        for div in article_divs(soup):
            number = int(div["id"].removeprefix("art_"))
            ti = div.find("p", class_="oj-ti-art")
            assert ti is not None and chars(ti.get_text("")) == chars(f"Article {number}"), (act.celex, number)
            title_text = by_number[number]["title"]
            original = chars(div.get_text(""))
            rebuilt = chars(by_number[number]["text"])
            assert original == rebuilt.replace(chars(header(number, title_text)), chars(f"Article {number}") + chars(title_text), 1), (act.celex, number)

    # Ventanas: pertenecen a un solo artículo, empiezan por la cabecera y cubren todas sus palabras.
    per_article: dict[tuple[str, int], list[dict]] = {}
    for w in windows:
        per_article.setdefault((w["celex"], w["article"]), []).append(w)
    for a in articles:
        wins = per_article[(a["celex"], a["article"])]
        head = header(a["article"], a["title"])
        assert all(w["text"].startswith(head + "\n") for w in wins)
        assert wins[0]["word_start"] == 0 and wins[-1]["word_end"] == a["n_words"]
        covered = set()
        for w in wins:
            covered |= set(range(w["word_start"], w["word_end"]))
        assert len(covered) == a["n_words"]
    assert len({w["chunk_id"] for w in windows}) == len(windows)  # ids únicos


@real_data
@pytest.mark.skipif(not ARTICLES_FILE.exists() or not WINDOWS_FILE.exists(), reason="chunks no generados")
def test_committed_chunks_match_regeneration():
    articles, windows = build_all()
    assert ARTICLES_FILE.read_text(encoding="utf-8") == dump_jsonl(articles)
    assert WINDOWS_FILE.read_text(encoding="utf-8") == dump_jsonl(windows)
    assert all(json.loads(line)["strategy"] == "article" for line in ARTICLES_FILE.read_text(encoding="utf-8").splitlines())
