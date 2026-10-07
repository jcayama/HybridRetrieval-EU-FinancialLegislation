import pytest

from scripts.golden import load_jsonl, stem, validate, words

ARTICLES = {
    ("C1", 1): {"celex": "C1", "article": 1, "title": "Alpha", "text": "Article 1. Alpha\nThe widget supervisor shall report every failure within twenty four hours."},
    ("C1", 2): {"celex": "C1", "article": 2, "title": "Beta", "text": "Article 2. Beta\nThe widget supervisor may also inspect premises."},
    ("C1", 3): {"celex": "C1", "article": 3, "title": "Gamma", "text": "Article 3. Gamma\nNothing relevant here."},
}


def q(qid, qtype, text):
    return {"query_id": qid, "type": qtype, "theme": "t", "query": text}


def r(qid, article, grade, celex="C1"):
    return {"query_id": qid, "celex": celex, "article": article, "grade": grade}


def run(queries, qrels):
    return validate(queries, qrels, ARTICLES)


def test_valid_set_has_no_errors():
    errors, _, _ = run([q("q001", "paraphrase", "who must tell the authority when things break")], [r("q001", 1, 3)])
    assert errors == []


def test_errors_are_detected():
    queries = [q("q001", "paraphrase", "a b c"), q("q001", "paraphrase", "x"), q("q002", "weird", "y"), q("q003", "exact_term", "z")]
    qrels = [r("q001", 1, 4), r("q001", 1, 3), r("q001", 99, 3), r("q009", 1, 3), r("q002", 1, True)]
    text = "\n".join(run(queries, qrels)[0])
    for expected in ("duplicado", "grado 4", "juicio duplicado", "no existe en el corpus", "la consulta no existe", "tipo 'weird'", "q003: sin ningún juicio", "grado True"):
        assert expected in text


def test_unexpected_keys_are_errors():
    errors, _, _ = run([q("q001", "paraphrase", "x")], [{"query_id": "q001", "celex": "C1", "article": [1, 2], "grade": [3, 2]}])
    assert any("sin listas" in e for e in errors)


def test_paraphrase_copying_a_phrase_warns():
    _, warnings, _ = run([q("q001", "paraphrase", "does the widget supervisor shall report matter")], [r("q001", 1, 3)])
    assert any("comparte la secuencia" in w for w in warnings)


def test_exact_term_warnings_and_info():
    queries = [q("q001", "exact_term", "widget supervisor")]
    _, warnings, info = run(queries, [r("q001", 1, 3), r("q001", 3, 1)])
    assert any("art. 3" in w for w in warnings)  # el término no está en el art. 3 juzgado
    assert any("art.2" in i for i in info)  # aparece en el art. 2, sin juicio


def test_multi_article_with_single_article_warns_and_no_grade3_warns():
    _, warnings, _ = run([q("q001", "multi_article", "x y z")], [r("q001", 1, 2)])
    text = "\n".join(warnings)
    assert "solo tiene 1 artículo" in text and "ningún artículo con grado 3" in text


def test_words():
    assert words("Lead Overseer, 24-hours!") == ["lead", "overseer", "24", "hours"]


def test_load_jsonl_reports_line_number(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n\n{mal}\n')
    with pytest.raises(ValueError, match=r"x\.jsonl:3"):
        load_jsonl(path)


def test_exact_term_tolerates_plurals_and_gerunds():
    articles = {("C1", 1): {"celex": "C1", "article": 1, "title": "A", "text": "Article 1. A\nRisk is monitored through appropriate stress testing procedures."}}
    _, warnings, _ = validate([q("q001", "exact_term", "stress tests")], [r("q001", 1, 3)], articles)
    assert not any("no aparece" in w for w in warnings)
    assert stem("tests") == stem("testing") == "test"
