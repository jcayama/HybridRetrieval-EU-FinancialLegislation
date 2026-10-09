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
| `python cli.py chunk` | Genera los chunks a partir del corpus congelado (comprueba antes los hashes): uno por artículo y ventanas de 200 palabras con solape de 40. Escribe `data/chunks/*.jsonl`. |
| `python cli.py list-articles [--act dora\|aifmd\|aiact] [--range 17-23]` | Lista artículos (acto, número, título y nº de palabras). |
| `python cli.py show-article <acto> <N>` | Muestra el texto completo de un artículo, tal y como se indexa. |
| `python cli.py validate-golden` | Valida el golden set: formato, que los artículos existen, grados 1–3, consultas sin juicios; muestra avisos de revisión y el reparto por tipo. Falla si hay errores o si el golden set congelado se modificó. |
| `python cli.py freeze-golden` | Congela el golden set (SHA-256 en `data/golden/golden.sha256`). Solo con el conjunto completo. |

### Datos

- `data/raw/`: HTML descargado de EUR-Lex. **No se versiona**: se vuelve a descargar y se valida
  con el hash.
- `data/corpus.sha256`: hashes congelados del corpus. Sí se versiona; es lo que garantiza que
  todos trabajan con el mismo texto.
- `data/chunks/`: chunks en JSONL (`articles.jsonl` y `windows_w200_o40.jsonl`). Sí se versionan.
  Un test comprueba que regenerarlos desde el corpus congelado da exactamente los mismos ficheros.

### Golden set

- `data/golden/queries.jsonl`: consultas (`query_id`, `type`, `theme`, `query`). Tipos: `paraphrase`,
  `exact_term` y `multi_article`. Las consultas están en inglés, como el corpus.
- `data/golden/qrels.jsonl`: juicios de relevancia, **una línea por par consulta-artículo**
  (`query_id`, `celex`, `article`, `grade`). Los escribe una persona; ningún LLM etiqueta.
- **Escala:** 3 = el artículo responde directamente; 2 = aporta una parte sustancial; 1 = relacionado,
  pero tangencial; lo no escrito = no relevante. El grado mide cuánto cubre ese artículo por sí solo,
  no el orden en que deba aparecer; puede haber varios artículos con el mismo grado.
- **Criterio de inclusión:** un artículo recibe juicio si **trata** el concepto de la consulta, no por
  mencionarlo de pasada. Por eso hay artículos que contienen el término de una consulta `exact_term`
  y no tienen juicio; `validate-golden` los lista como información para revisión.
- **Método:** consultas escritas «desde el artículo», juzgadas leyendo el texto, sin ayuda de ningún
  buscador, y congeladas antes de ejecutar ninguna recuperación.
- **Limitaciones:** conjunto pequeño (30–50 consultas) y un solo anotador; el método «desde el artículo»
  favorece ligeramente el acierto del artículo de origen. Las diferencias pequeñas entre sistemas no
  serán estadísticamente sólidas.

### Chunking

- **Alcance:** solo artículos (248: AIFMD 71, DORA 64, AI Act 113). Considerandos y anexos quedan fuera.
- **Por artículo:** un chunk por artículo, con cabecera `Article N. Título`.
- **Por ventanas:** 200 palabras del cuerpo con solape de 40, sin cruzar el límite de un artículo; si la
  cola nueva tiene menos de 50 palabras se fusiona con la ventana anterior. Cada ventana empieza por la
  cabecera `Article N. Título` (sus palabras no cuentan para las 200).
- **Fidelidad:** el texto es literal; solo se normalizan espacios. Las filas de las listas `(a) ...` se
  unen en una línea.
- **Evaluación:** los juicios de relevancia se darán por artículo (`celex` + `article`), de modo que
  ambas estrategias se miden con los mismos juicios.

## Evaluación

Las métricas viven en `scripts/metrics.py` como funciones puras, con tests hechos a mano
(`tests/test_metrics.py`). Decisiones fijadas **antes** de ejecutar ninguna recuperación:

- **Unidad:** el artículo. Los rankings de chunks (ventanas) se convierten a artículos distintos,
  conservando la mejor posición de cada uno.
- **Relevante** (Recall y MRR): grado >= 2. El grado 1 solo cuenta en nDCG.
- **Métricas:** Recall@1, @3, @5 y @10, MRR@10 y nDCG@10 con ganancia lineal (3, 2, 1). La ganancia
  exponencial (7, 3, 1) queda como análisis de sensibilidad.
- **Incertidumbre:** bootstrap pareado sobre consultas (10.000 remuestras, semilla fija, intervalo
  percentil del 95 %), también por tipo de consulta. Con 15, 15 y 10 consultas por tipo, los intervalos
  serán anchos.

Ejemplo hecho a mano (juicios A=3, B=2, C=1; ranking X, B, A, Y, C con X e Y irrelevantes):
Recall@1 = 0, Recall@2 = 0,5, Recall@3 = 1, MRR = 0,5 y nDCG@5 = 0,6612
(DCG = 2/log2(3) + 3/log2(4) + 1/log2(6) = 3,1487; ideal A, B, C = 4,7619).

## Atribución y licencia del corpus

Los textos legislativos proceden de [EUR-Lex](https://eur-lex.europa.eu/), © Unión Europea,
y se reutilizan conforme a la Decisión 2011/833/UE de la Comisión, citando la fuente y sin
distorsionar el texto.

<!-- Pendiente: confirmar el aviso legal de EUR-Lex antes de publicar; licencia del código. -->
