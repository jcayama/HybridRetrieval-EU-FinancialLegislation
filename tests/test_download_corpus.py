from scripts.download_corpus import count_articles, read_hashes, sha256_bytes, write_hashes

HTML = """
<html><body>
  <div class="eli-subdivision" id="art_1"><p class="oj-ti-art">Article 1</p></div>
  <div class="eli-subdivision" id="art_2"><p class="oj-ti-art">Article 2</p></div>
  <div class="eli-subdivision" id="art_2"><p>repetido</p></div>
  <div class="eli-subdivision" id="rct_1"><p>considerando, no es artículo</p></div>
  <div class="eli-subdivision" id="art_3.a"><p>id no numérico</p></div>
  <div id="art_9"><p>sin la clase eli-subdivision</p></div>
</body></html>
"""


def test_count_articles_counts_unique_numeric_article_ids():
    assert count_articles(HTML) == 2


def test_count_articles_accepts_bytes():
    assert count_articles(HTML.encode()) == 2


def test_count_articles_empty_document():
    assert count_articles("<html></html>") == 0


def test_sha256_known_value():
    # SHA-256 de "abc" (vector de prueba estándar de NIST)
    assert sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_hashes_roundtrip(tmp_path):
    path = tmp_path / "corpus.sha256"
    hashes = {"b.html": "bb", "a.html": "aa"}
    write_hashes(hashes, path)
    assert path.read_text() == "aa  a.html\nbb  b.html\n"
    assert read_hashes(path) == hashes


def test_read_hashes_missing_file_returns_empty(tmp_path):
    assert read_hashes(tmp_path / "no_existe") == {}
