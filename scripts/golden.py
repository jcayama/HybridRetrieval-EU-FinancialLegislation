"""Validación y congelado del golden set (data/golden/queries.jsonl y qrels.jsonl).

Los juicios de relevancia (qrels) los escribe una persona; este módulo solo comprueba su forma y
señala cosas que conviene revisar, sin proponer grados.
"""

import json
import re
import sys
from pathlib import Path

from scripts.browse_corpus import load_articles
from scripts.download_corpus import ROOT, read_hashes, sha256_bytes, write_hashes

GOLDEN_DIR = ROOT / "data" / "golden"
QUERIES_FILE = GOLDEN_DIR / "queries.jsonl"
QRELS_FILE = GOLDEN_DIR / "qrels.jsonl"
HASH_FILE = GOLDEN_DIR / "golden.sha256"

TYPES = ("paraphrase", "exact_term", "multi_article")
QUOTAS = {"paraphrase": 15, "exact_term": 15, "multi_article": 10}
GRADES = (1, 2, 3)
QUERY_KEYS = {"query_id", "type", "theme", "query"}
QREL_KEYS = {"query_id", "celex", "article", "grade"}
COPY_NGRAM = 5  # una paráfrasis no debería compartir ninguna secuencia de 5 palabras con un artículo relevante


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def stem(token: str) -> str:
    """Raíz tosca (quita -ing, -es, -s) para que «stress tests» encaje con «stress testing»."""
    for suffix in ("ing", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def stemmed(tokens: list[str]) -> str:
    return " ".join(stem(t) for t in tokens)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path.name}:{number}: JSON inválido ({error})") from error
    return rows


def _shares_ngram(query: list[str], text: list[str], n: int) -> str | None:
    grams = {" ".join(text[i : i + n]) for i in range(len(text) - n + 1)}
    for i in range(len(query) - n + 1):
        gram = " ".join(query[i : i + n])
        if gram in grams:
            return gram
    return None


def validate(queries: list[dict], qrels: list[dict], articles: dict[tuple[str, int], dict]) -> tuple[list[str], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    info: list[str] = []

    ids = [q.get("query_id") for q in queries]
    for qid in {i for i in ids if ids.count(i) > 1}:
        errors.append(f"{qid}: id de consulta duplicado")
    by_id = {q["query_id"]: q for q in queries if "query_id" in q}
    for q in queries:
        qid = q.get("query_id", "?")
        if set(q) != QUERY_KEYS:
            errors.append(f"{qid}: claves {sorted(q)} (se esperaban {sorted(QUERY_KEYS)})")
        if q.get("type") not in TYPES:
            errors.append(f"{qid}: tipo {q.get('type')!r} no válido (usa {', '.join(TYPES)})")
        if not str(q.get("query", "")).strip():
            errors.append(f"{qid}: consulta vacía")

    seen: set[tuple] = set()
    relevant: dict[str, list[dict]] = {qid: [] for qid in by_id}
    for r in qrels:
        label = f"{r.get('query_id')}/{r.get('celex')}/art.{r.get('article')}"
        if set(r) != QREL_KEYS:
            errors.append(f"{label}: claves {sorted(r)} (se esperaban {sorted(QREL_KEYS)})")
            continue
        if not (isinstance(r["query_id"], str) and isinstance(r["celex"], str) and isinstance(r["article"], int) and not isinstance(r["article"], bool)):
            errors.append(f"{label}: query_id y celex deben ser texto y article un entero (una línea por artículo, sin listas)")
            continue
        key = (r["query_id"], r["celex"], r["article"])
        if r["query_id"] not in by_id:
            errors.append(f"{label}: la consulta no existe en queries.jsonl")
            continue
        if not isinstance(r["grade"], int) or isinstance(r["grade"], bool) or r["grade"] not in GRADES:
            errors.append(f"{label}: grado {r['grade']!r} no válido (usa 1, 2 o 3)")
        if (r["celex"], r["article"]) not in articles:
            errors.append(f"{label}: el artículo no existe en el corpus")
        if key in seen:
            errors.append(f"{label}: juicio duplicado")
        seen.add(key)
        relevant[r["query_id"]].append(r)

    for qid, q in by_id.items():
        rels = relevant[qid]
        if not rels:
            errors.append(f"{qid}: sin ningún juicio de relevancia")
            continue
        if not any(r["grade"] == 3 for r in rels if isinstance(r["grade"], int)):
            warnings.append(f"{qid}: ningún artículo con grado 3 (¿es intencionado?)")
        texts = {(r["celex"], r["article"]): words(articles[(r["celex"], r["article"])]["text"]) for r in rels if (r["celex"], r["article"]) in articles}
        qwords = words(q.get("query", ""))
        if q.get("type") == "multi_article" and len(rels) < 2:
            warnings.append(f"{qid}: es multi_article pero solo tiene {len(rels)} artículo relevante")
        if q.get("type") == "paraphrase":
            for key, text in texts.items():
                gram = _shares_ngram(qwords, text, COPY_NGRAM)
                if gram:
                    warnings.append(f"{qid}: la paráfrasis comparte la secuencia «{gram}» con el art. {key[1]} de {key[0]}")
                    break
        if q.get("type") == "exact_term" and qwords:
            phrase = stemmed(qwords)
            for key, text in texts.items():
                if phrase not in stemmed(text):
                    warnings.append(f"{qid}: el término «{q['query']}» no aparece en el texto del art. {key[1]} de {key[0]}")
            judged = set(texts)
            others = [f"{a['celex']}/art.{a['article']}" for k, a in articles.items() if k not in judged and phrase in stemmed(words(a["text"]))]
            if others:
                info.append(f"{qid}: el término aparece también en artículos sin juicio: {', '.join(others)} (revisa si deben tenerlo)")

    counts = {t: sum(1 for q in queries if q.get("type") == t) for t in TYPES}
    info.append("reparto: " + ", ".join(f"{t} {counts[t]}/{QUOTAS[t]}" for t in TYPES) + f" (total {len(queries)}; objetivo 30–50)")
    return errors, warnings, info


def hashes_of_files() -> dict[str, str]:
    return {f.name: sha256_bytes(f.read_bytes()) for f in (QUERIES_FILE, QRELS_FILE)}


def check_frozen() -> list[str]:
    """Si el golden set está congelado, comprueba que no se ha modificado."""
    frozen = read_hashes(HASH_FILE)
    if not frozen:
        return []
    current = hashes_of_files()
    return [f"{name}: modificado después de congelar (SHA-256 distinto de golden.sha256)" for name, digest in frozen.items() if current.get(name) != digest]


def _run_validation() -> tuple[list[str], list[str], list[str]]:
    queries, qrels = load_jsonl(QUERIES_FILE), load_jsonl(QRELS_FILE)
    articles = {(a["celex"], a["article"]): a for a in load_articles()}
    return validate(queries, qrels, articles)


def run_validate() -> int:
    try:
        errors, warnings, info = _run_validation()
    except (ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    errors += check_frozen()
    for title, items in (("ERRORES", errors), ("AVISOS (revisar, no son fallos)", warnings), ("INFO", info)):
        if items:
            print(f"\n{title}")
            print("\n".join(f"  - {i}" for i in items))
    print("\nGolden set válido." if not errors else f"\n{len(errors)} error(es).")
    return 1 if errors else 0


def run_freeze() -> int:
    try:
        errors, _, _ = _run_validation()
    except (ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if errors:
        print("ERROR: hay errores de validación; no se congela. Ejecuta `python cli.py validate-golden`.", file=sys.stderr)
        return 1
    write_hashes(hashes_of_files(), HASH_FILE)
    print(f"Golden set congelado en {HASH_FILE.relative_to(ROOT)}")
    return 0
