# Evaluación de recuperación híbrida sobre legislación financiera de la UE

Mini-proyecto que compara estrategias de recuperación sobre Postgres + pgvector y las evalúa
con un golden set congelado. Todas las cifras de este README saldrán de scripts reproducibles
del repositorio.

> **Estado: en construcción.** Todavía no hay resultados publicados.

## Qué se evaluará

- Chunking por artículo frente a ventanas fijas.
- Búsqueda de texto completo (FTS de Postgres) frente a búsqueda densa (pgvector) frente a
  híbrido con Reciprocal Rank Fusion (RRF).
- Métricas: Recall@K, MRR y nDCG@10 sobre un golden set congelado de 30–50 consultas
  (paráfrasis, término exacto y multi-artículo con relevancia graduada).
- Índices HNSW frente a IVFFlat (recall y latencia).
- Aislamiento multi-inquilino con Row-Level Security (RLS) y un test que falla ante cualquier fuga.
- GitHub Actions que falla si Recall@K cae por debajo de un umbral o si el test de RLS detecta
  una fuga.

## Corpus

Texto publicado (no consolidado) de tres actos de la UE, descargado de la API Cellar de EUR-Lex:

| Acto | CELEX |
|------|-------|
| Directiva AIFMD 2011/61/UE | 32011L0061 |
| Reglamento DORA (UE) 2022/2554 | 32022R2554 |
| Reglamento de IA (UE) 2024/1689 | 32024R1689 |

El corpus se congela guardando el SHA-256 de cada descarga y versionando los chunks en JSONL.

## Datos y privacidad

Solo se usan datos públicos o sintéticos. No hay datos de clientes en este repositorio.

## Desarrollo

Requiere Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install   # imprescindible: activa el detector de secretos
cp .env.example .env # y rellena tus valores; .env nunca se versiona
```

El hook de pre-commit ejecuta [gitleaks](https://github.com/gitleaks/gitleaks), detecta claves
privadas y bloquea cualquier fichero `.env` (salvo `.env.example`).

Los tests se ejecutan con `pytest` y el estilo se comprueba con `ruff check .`.

## Comandos

Todas las etapas del pipeline se lanzan desde un único punto de entrada, `cli.py`, desde la raíz
del repositorio y con el entorno virtual activo. `python cli.py --help` lista los comandos
disponibles. Se irán añadiendo conforme se implementen las etapas.

| Comando | Qué hace |
|---------|----------|
| `python cli.py download-corpus` | Descarga el corpus y **verifica** que el SHA-256 de cada acto coincide con `data/corpus.sha256` y que el número de artículos es el esperado (71, 64 y 113). Falla si algo no coincide. |
| `python cli.py download-corpus --freeze` | Igual, pero **escribe** `data/corpus.sha256`. Solo se usa la primera vez, para congelar el corpus. |

### Datos

- `data/raw/`: HTML descargado de EUR-Lex. **No se versiona**: se vuelve a descargar y se valida
  con el hash.
- `data/corpus.sha256`: hashes congelados del corpus. Sí se versiona; es lo que garantiza que
  todos trabajan con el mismo texto.

## Atribución y licencia del corpus

Los textos legislativos proceden de [EUR-Lex](https://eur-lex.europa.eu/), © Unión Europea,
y se reutilizan conforme a la Decisión 2011/833/UE de la Comisión, citando la fuente y sin
distorsionar el texto.

<!-- Pendiente: confirmar el aviso legal de EUR-Lex antes de publicar; licencia del código. -->
