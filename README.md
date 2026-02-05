# Coal Expert KB — Expert Knowledge Base for Coal Pyrolysis & Gasification
**RAG + Expert Metadata + Structured Records + (Optional) LoRA/QLoRA**

Coal Expert KB builds an **auditable, metadata-aware expert knowledge base** from scientific PDFs and documents on coal pyrolysis/gasification, with special focus on **pollutant formation** under different operating conditions.

- **Nitrogen pollutants**: NH₃, HCN, NOx  
- **Sulfur pollutants**: H₂S, SO₂, COS  
- **Aromatics / phenolics**: benzene, phenol, etc.

This repository is designed for two end goals:

1) **Expert retrieval & QA**: “Filter by operating conditions, then answer with evidence.”  
2) **Trainable datasets**: extract structured `ExperimentRecord`s for downstream modeling (prediction, LoRA extraction models, etc.)

> Out of the box, the default config uses **Alibaba Cloud Bailian / DashScope (OpenAI-compatible mode)** via LangChain for embeddings/LLM.

---

## Contents

- [What this repo does](#what-this-repo-does)
- [How it works](#how-it-works)
- [Quickstart (first-time user)](#quickstart-first-time-user)
- [Choose a backend: Elasticsearch vs Chroma](#choose-a-backend-elasticsearch-vs-chroma)
- [DashScope / Bailian setup](#dashscope--bailian-setup)
- [Core commands](#core-commands)
- [Configuration guide](#configuration-guide)
- [Structured records (SQLite)](#structured-records-sqlite)
- [Evaluation](#evaluation)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

---

## What this repo does

A plain “chat with PDFs” vector DB is usually not enough for scientific work. This repo adds:

1) **Constraint-aware retrieval (hard + soft filters)**
   - Example query: “steam gasification, 1200–1400 K, 2–3 MPa, NH₃ + HCN”
   - The system tries to **reliably narrow** results by stage/gas/targets/T/P and still returns useful evidence when metadata is missing.

2) **Auditability**
   - Extracted fields can attach evidence spans/snippets so you can verify quickly.
   - Each retrieved chunk includes file/page/section for citation.

3) **Structured, trainable outputs**
   - Extract `ExperimentRecord`s with normalization and conflict tracking.
   - Export clean CSV/JSONL for modeling or fine-tuning.

---

## How it works

```text
PDFs / Docs (data/raw_pdfs/, data/raw_docs/)
   │
   ├─► Parse (PDF -> pages cache) + clean text
   │
   ├─► Chunking (section-aware splitter)
   │
   ├─► Metadata extraction
   │     ├─ rules: range/ratio/targets/stage + normalization
   │     └─ optional LLM augmentation (fill missing only)
   │
   ├─► Indexing backend
   │     ├─ Elasticsearch (recommended)  OR
   │     └─ Chroma (local fallback)
   │
   ├─► Retrieval (filters + ranking: vector + optional BM25 + soft constraints)
   │
   └─► (Optional) QA generation with citations (RAG)
         + (Optional) Record extraction → SQLite → Export datasets
```

Storage (default paths; configurable in `configs/app.yaml`):

- **Pages cache**: `data/interim/`
- **Registry DB** (runs/documents/chunks/query logs): `storage/kb.db`
- **Records DB** (`ExperimentRecord`s): `storage/expert.db`
- **Chroma** persistence (when backend=chroma/both): `storage/chroma_db/`
- **Elasticsearch** data (when using docker compose): docker volume `elasticsearch_data`

---

## Quickstart (first-time user)

### 0) Requirements

- Python 3.10+ recommended
- macOS / Linux / WSL supported
- (If using Elasticsearch locally) Docker + docker compose

### 1) Install

```bash
pip install -e .[dev]
pytest -q
```

If you plan to use DashScope/Bailian for embeddings/LLM (recommended):

```bash
pip install -U langchain-openai openai python-dotenv
```

### 2) Add documents

Put your PDFs/docs under:

```text
data/raw_pdfs/
data/raw_docs/
```

Supported formats: pdf, txt, md, html, docx, pptx, csv, xlsx, json, jsonl.

> Optional loaders (html/docx/pptx/xlsx) may require extras: `pip install -e .[docs]`.

### 3) Configure API key (DashScope/Bailian)

Create `.env` at repo root:

```env
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
```

(Optional) Or export it in your shell:

```bash
export DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 4) Start Elasticsearch (recommended backend)

```bash
docker compose up -d
curl -s http://localhost:9200
```

### 5) Build a clean, versioned index (recommended for first run)

```bash
python scripts/index.py build --embedding-version v1
```

Why this is recommended:
- creates a new index name with version + schema hash
- ingests **from scratch**
- validates the index
- switches `alias_current` only after validation

### 6) Ask questions

```bash
python scripts/ask.py --backend elastic --mode balanced
```

## Example queries (for experiments)

下面这些 query 专门用来做实验：验证 **ontology 解析（stage / gas_agent / pollutants）**、温度/压力范围解析、strict/broad 模式差异、以及 rerank 的影响。

~~~bash
# 推荐先用 elastic（默认开启rerank，可过滤字段更强）
python scripts/ask.py --backend elastic --mode balanced

# 对比模式（同一批 queries 分别跑一遍）
# python scripts/ask.py --backend elastic --mode strict
# python scripts/ask.py --backend elastic --mode broad

# 可选：开启 LLM answer（检索不变，回答方式变）
# python scripts/ask.py --backend elastic --mode balanced --llm
~~~

### A) Baseline（无显式约束：看语义召回）
- NH3 formation mechanism in coal gasification (with evidence)
- HCN formation pathways in coal pyrolysis (cite pages/sections)
- 氮氧化物 形成机理 与 NH3/HCN 的关系（给证据）

### B) Stage + Pollutant（验证 stage / pollutant 解析）
- 气化 NH3 生成机理
- 热解 HCN 生成来源
- combustion NOx 形成（证据）
- ignition NOx 前驱体（NH3/HCN）证据

### C) Stage + Gas agent + Pollutant（验证 gas_agent 解析）
- 蒸汽 气化 NH3 HCN
- H2O gasification ammonia NH3
- CO2 气化 氰化氢 HCN
- carbon dioxide gasification HCN
- air combustion NOx
- 氮气 气化 NH3（对比惰性气氛）

### D) 加温度范围（验证 1100-1300K / 800-950°C 等解析）
- 蒸汽 气化 NH3 1100-1300K
- CO2 气化 HCN 1200-1400K
- 热解 HCN 800-950°C
- pyrolysis phenols 600-900 C
- combustion NOx 900-1200K

### E) 加压力范围（验证 MPa 区间解析）
- 加压 气化 蒸汽 NH3 2-3 MPa
- CO2 气化 HCN 0.5-2 MPa
- steam gasification NH3 1 MPa 1200K

### F) 对比型问题（更贴近论文检索：看证据质量 + 多样性）
- 蒸汽气化 vs CO2气化：NH3 与 HCN 的差异与证据对比
- 温度升高（1100→1400K）对 NH3/HCN 哪个更敏感？请给证据
- 热解与气化阶段：phenols 与 benzene 的生成差异（引用证据）

### G) “严格措辞”压力测试（更容易触发 hard constraint）
- 只考虑 蒸汽 气化，温度必须 1200-1400K，目标 NH3 和 HCN
- 必须是 CO2 气化，压力 2-3 MPa，关注 HCN
- 只看 热解，不要 气化：HCN 的来源证据

> 实验建议：
> 1) 同一组 queries 跑 strict/balanced/broad，观察召回与约束满足度变化  
> 2) 开/关 rerank 对比 top-3 证据质量  
> 3) 记录 ask.py 输出的 “解析到的约束” 与 “Retrieval trace” 做实验日志

Enable LLM answer generation (still evidence-grounded):

```bash
python scripts/ask.py --backend elastic --llm
```

---

## Choose a backend: Elasticsearch vs Chroma

This repo supports three modes via `configs/app.yaml` → `backend`:

- `elastic` (default, recommended)
- `chroma` (local fallback)
- `both` (index into both; can retrieve from both)

### Elasticsearch (`backend: elastic`) — recommended

Pros:
- robust filtering at scale
- index versioning + aliases (`*_current`, `*_prev`)
- good observability (Kibana)

Typical workflow:
- **First run / major changes**: `python scripts/index.py build --embedding-version vX`
- **Daily incremental updates**: `python scripts/ingest.py`

> Important: `python scripts/ingest.py` (when backend=elastic) is an *ingestion tool* and does **not** perform the strict “build → validate → switch alias” lifecycle that `index.py build` performs. For production-like usage, use `index.py build` at least once.

### Chroma (`backend: chroma`) — local fallback

Pros:
- simplest local setup (no ES)

Usage:
1) Set `backend: chroma` in `configs/app.yaml`
2) Rebuild & ingest:

```bash
rm -rf storage/chroma_db storage/manifest.json
python scripts/ingest.py --rebuild
```

### Both (`backend: both`)

- Ingest writes to both Chroma and Elasticsearch.
- Query-time retrieval can fuse results (RRF).
- Good for debugging/ablation.

---

## DashScope / Bailian setup

This repo supports Alibaba Cloud Bailian (DashScope) via **OpenAI-compatible mode**.

In `configs/app.yaml` (defaults shown):

```yaml
llm:
  provider: dashscope
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  api_key_env: "DASHSCOPE_API_KEY"
  model: "qwen-plus"
  temperature: 0
  timeout: 60

embeddings:
  provider: dashscope
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  api_key_env: "DASHSCOPE_API_KEY"
  model: "text-embedding-v4"
  dimensions: 1024
```

### 10-second smoke tests (recommended)

Embeddings:

```bash
python -c "from coal_kb.embeddings.factory import EmbeddingsConfig, make_embeddings; cfg=EmbeddingsConfig(provider='dashscope', base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', api_key_env='DASHSCOPE_API_KEY', model='text-embedding-v4', dimensions=1024); emb=make_embeddings(cfg); v=emb.embed_query('NH3 HCN formation'); print('dim=', len(v))"
```

Chat LLM:

```bash
python -c "from coal_kb.llm.factory import LLMConfig, make_chat_llm; cfg=LLMConfig(provider='dashscope', base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', api_key_env='DASHSCOPE_API_KEY', model='qwen-plus', temperature=0, timeout=60); llm=make_chat_llm(cfg); print(llm.invoke('用一句话解释煤气化。').content)"
```

---

## Core commands

### 1) Ingest (`scripts/ingest.py`)

```bash
python scripts/ingest.py
```

Options:

- `--rebuild`  
  Clears `storage/manifest.json` and rebuilds from scratch.  
  (Note: rebuild also clears `storage/chroma_db/` even if you are using elastic-only; safe but surprising.)

- `--force`  
  Continue even if some batches fail, or if you intentionally want to bypass certain safety checks (not recommended).

- `--tables` / `--table-flavor`  
  Enable PDF table extraction (requires `camelot-py`).

- `--llm-metadata`  
  Enable LLM augmentation for metadata extraction (fills missing fields only).

**Important: manifest signature mismatch**
- If you change `configs/app.yaml` in a way that changes embedding/chunking/schema signatures, ingestion will **stop with an error** and tell you to use `--rebuild` or `--force`.
- This is intentional to avoid mixing stale embeddings/chunks.

### 2) Build ES index (versioned & validated) (`scripts/index.py`)

Recommended for:
- first-time setup
- changing embedding model/dimensions
- changing schema or chunking significantly
- “clean rebuild” to avoid incremental drift

```bash
python scripts/index.py build --embedding-version v1
```

Other commands:

```bash
python scripts/index.py rollback
python scripts/index.py switch --index <full_index_name>
```

### 3) Ask (`scripts/ask.py`)

```bash
python scripts/ask.py --backend elastic --mode balanced
```

Useful flags:
- `--llm` enable answer generation (requires `llm` config)
- `--rerank` enable reranking (can also be enabled in config)
- `--mode strict|balanced|broad` adjust constraint strictness
- `--k` override top-k

### 4) Validate ES index (optional but recommended after big changes)

If your repo includes `scripts/validate_index.py`, run it after build:

```bash
python scripts/validate_index.py --index coal_kb_chunks_current
```

(If you don’t have the script in your checkout, you can still verify via `/_cat/indices` and a simple search query in Kibana.)

---

## Configuration guide

Main config: `configs/app.yaml`.

Common edits:

- `paths.*`: where PDFs/outputs/DBs are stored
- `backend`: `elastic` | `chroma` | `both`
- `chunking.*`: chunk size/overlap + `profile_by_section`
- `retrieval.*`: k/candidates/rerank/diversity/max_relax_steps/mode
- `elastic.*`: host, index prefix, aliases, bulk chunk size, ICU analyzer
- `model_versions.embedding_version`: label used for ES index naming
- `llm.*` and `embeddings.*`: DashScope/OpenAI-compatible provider config

### When you MUST rebuild

You should rebuild if you change any of:
- embeddings model or dimensions
- chunking sizes/profiles
- schema ontology (`configs/schema.yaml`)

Rebuild (Elastic, recommended):

```bash
python scripts/index.py build --embedding-version v2
```

Rebuild (Chroma):

```bash
rm -rf storage/chroma_db storage/manifest.json
python scripts/ingest.py --rebuild
```

---

## Structured records (SQLite)

Record extraction writes to: `storage/expert.db`

### Extract records

```bash
python scripts/extract_records.py --limit 300
```

Enable LLM extraction (more accurate but costs money):

```bash
python scripts/extract_records.py --llm --limit 300
```

### Export records

```bash
python scripts/export_records.py --out data/artifacts/records.csv
```

What you get:
- normalized operating condition fields (stage/gas/T/P/ratios/coal_name)
- pollutant outputs (value + unit + basis when available)
- evidence quote/snippet
- conflict flags if inconsistent outputs appear under the same signature

---

## Evaluation

Two common evaluation entrypoints:

### 1) End-to-end eval (`scripts/eval.py`)

```bash
python scripts/eval.py --gold data/eval/eval_set.jsonl
```

### 2) Retrieval-focused eval (`scripts/eval_retrieval.py`)

```bash
python scripts/eval_retrieval.py --gold data/eval/retrieval_gold.jsonl --k 5
```

Metrics typically include Recall@K, FilterPrecision@K, Diversity@K, and ReferencesHit@K.

---

## Project structure

```text
coal-expert-kb/
  configs/
    app.yaml
    schema.yaml
    prompts/
  data/
    raw_pdfs/
    raw_docs/
    interim/
    artifacts/
  storage/
    chroma_db/      # when backend=chroma/both
    kb.db           # registry
    expert.db       # ExperimentRecord
    manifest.json
  src/
    coal_kb/
      settings.py
      pipelines/
      parsing/
      chunking/
      metadata/
      retrieval/
      store/
      qa/
  scripts/
    ingest.py
    index.py
    ask.py
    extract_records.py
    export_records.py
    eval.py
    eval_retrieval.py
  tests/
```

---

## Troubleshooting

### 1) “Manifest signature mismatch detected”
Cause: you changed embeddings/chunking/schema since the last ingest.

Fix (recommended):

```bash
python scripts/index.py build --embedding-version vX
```

Or if you are using Chroma:

```bash
rm -rf storage/chroma_db storage/manifest.json
python scripts/ingest.py --rebuild
```

### 2) “Elasticsearch is not reachable”
- Ensure `docker compose up -d` is running.
- Confirm `configs/app.yaml` has correct `elastic.host` (default `http://localhost:9200`).

### 3) “Embedding dimensions mismatch / mapping errors”
- Ensure `embeddings.dimensions` matches the real output dimension (run the smoke test).
- Rebuild ES index via `python scripts/index.py build --embedding-version vX`.

### 4) Chroma retrieval got worse after switching embedding model
Rebuild Chroma (old vectors are incompatible):

```bash
rm -rf storage/chroma_db storage/manifest.json
python scripts/ingest.py --rebuild
```

### 5) LLM cost control
- Start with rule-only ingest: don’t pass `--llm-metadata`.
- Enable LLM only when needed: `--llm-metadata` and/or `ask.py --llm`.
- Prefer `--mode balanced` (strict may drop useful evidence when metadata is missing).

---

## Roadmap

- Stronger section detection (Methods/Results/Table) to reduce LLM calls
- Better evidence span alignment for LLM-filled fields
- Table-first extraction (for pollutant numbers and operating conditions)
- Schema-constrained decoding for strict JSON extraction
- Active learning loop: conflicts → human review → curated pairs → retrain LoRA

---

## License

MIT or Apache-2.0 (choose one and keep it consistent)

### Elasticsearch ICU analyzer (optional)

If you enable `elastic.enable_icu_analyzer: true`, make sure your Elasticsearch has the ICU plugin installed. Otherwise index creation may fail.

For local dev via docker compose, you can either:
- disable ICU analyzer in config, or
- install the plugin in the ES image and rebuild the index.
