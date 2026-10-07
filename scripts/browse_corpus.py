"""Lectura del corpus en la terminal (sobre data/chunks/articles.jsonl), para redactar y juzgar el golden set."""

import json
import sys

from scripts.chunk_corpus import ARTICLES_FILE
from scripts.download_corpus import ACTS

ALIASES = {"aifmd": "32011L0061", "dora": "32022R2554", "aiact": "32024R1689"}
assert {act.celex for act in ACTS} == set(ALIASES.values())


def resolve_act(name: str) -> str:
    """Admite el alias (aifmd, dora, aiact) o el número CELEX."""
    key = name.lower()
    if key in ALIASES:
        return ALIASES[key]
    if name in ALIASES.values():
        return name
    raise ValueError(f"Acto desconocido: {name!r}. Usa uno de {sorted(ALIASES)} o un CELEX.")


def load_articles() -> list[dict]:
    if not ARTICLES_FILE.exists():
        raise FileNotFoundError(f"Falta {ARTICLES_FILE.name}; ejecuta `python cli.py chunk`.")
    return [json.loads(line) for line in ARTICLES_FILE.read_text(encoding="utf-8").splitlines()]


def parse_range(text: str) -> tuple[int, int]:
    low, _, high = text.partition("-")
    first, last = int(low), int(high or low)
    if first > last:
        raise ValueError(f"Rango inválido: {text!r}")
    return first, last


def list_articles(act: str | None = None, article_range: str | None = None) -> list[str]:
    celex = resolve_act(act) if act else None
    low, high = parse_range(article_range) if article_range else (0, 10**9)
    rows = []
    for a in load_articles():
        if (celex is None or a["celex"] == celex) and low <= a["article"] <= high:
            label = next(k for k, v in ALIASES.items() if v == a["celex"])
            rows.append(f"{label:<5} {a['article']:>3}  {a['title'][:95]}  [{a['n_words']} palabras]")
    return rows


def get_article(act: str, number: int) -> dict:
    celex = resolve_act(act)
    for a in load_articles():
        if a["celex"] == celex and a["article"] == number:
            return a
    raise KeyError(f"No existe el artículo {number} en {act}.")


def run_list(act: str | None, article_range: str | None) -> int:
    try:
        rows = list_articles(act, article_range)
    except (ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("\n".join(rows) if rows else "(sin resultados)")
    return 0


def run_show(act: str, number: int) -> int:
    try:
        article = get_article(act, number)
    except (ValueError, KeyError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"[{article['act']} · {article['celex']} · {article['chapter_id']} · {article['n_words']} palabras]\n")
    print(article["text"])
    return 0
