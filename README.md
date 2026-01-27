# Coal Expert KB — Expert Knowledge Base for Coal Pyrolysis & Gasification  
**RAG + Expert Metadata + Structured Records + (Optional) LoRA/QLoRA**

Coal Expert KB builds an **auditable, metadata-aware expert knowledge base** from scientific PDFs on coal pyrolysis/gasification, with special focus on **pollutant formation** under different operating conditions:

- **Nitrogen pollutants**: NH₃, HCN, NOx  
- **Sulfur pollutants**: H₂S, SO₂, COS  
- **Aromatics / phenolics**: benzene, phenol, etc.

This repository is designed for two end goals:
1) **Expert retrieval & QA**: “Filter by operating conditions, then answer with evidence.”  
2) **Trainable datasets**: extract structured `ExperimentRecord`s for downstream modeling (deep learning prediction, LoRA extraction models, etc.)

> Works with **Alibaba Cloud Bailian / DashScope (OpenAI-compatible mode)** out of the box via LangChain.

---

## Contents

- [Why this project](#why-this-project)
- [What you get](#what-you-get)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [DashScope / Bailian setup](#dashscope--bailian-setup)
- [Configuration guide](#configuration-guide)
- [Pipelines & workflows](#pipelines--workflows)
- [Evidence & metadata design](#evidence--metadata-design)
- [Structured records (SQLite)](#structured-records-sqlite)
- [LoRA/QLoRA fine-tuning (shortest path)](#loraqlora-fine-tuning-shortest-path)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Why this project

A plain “chat with PDFs” vector DB is not enough for scientific work because you need:

1) **Hard filtering by operating conditions**  
   Example: “steam gasification, 1200–1400 K, 2–3 MPa, NH₃ + HCN” should narrow results **reliably**.

2) **Auditability**  
   Every extracted condition/value should trace back to a **PDF chunk** (and ideally page), so you can verify quickly.

3) **Trainable structured outputs**  
   If you want predictive models or LoRA extractors, you need **clean records**, **unit normalization**, and **conflict tracking**.

---

## What you get

### 1) RAG retrieval with expert filters
- Filter fields (examples):
  - `stage`: pyrolysis / gasification / coupled
  - `gas_agent`: steam / O₂ / air / CO₂ / …
  - `targets`: NH₃, HCN, NOx, H₂S, SO₂, COS, benzene, phenol, …
  - `T_range_K`, `P_range_MPa` + overlap checks
  - optional `coal_name` matching
- Ranking:
  - vector similarity candidates
  - optional BM25 candidate rerank + RRF fusion (stable for chemical symbols/units)

### 2) Expert metadata extraction (rule-first + optional LLM)
- Rule-first extraction (fast, deterministic):
  - temperature/pressure **single + range**
  - ratios (S/C, ER, O/C, …)
  - targets detection + stage/gas normalization
- Evidence artifacts:
  - `meta_confidence`: field → confidence score
  - `meta_evidence`: field → evidence snippet/span
- Optional LLM augmentation (**provider-agnostic injection**):
  - fill missing fields only (conservative merge)
  - cost control: only call LLM when key fields are too missing

### 3) Structured records in SQLite (`ExperimentRecord`)
- Store trainable records in `storage/expert.db`
- Track:
  - unit normalization (when safe)
  - evidence quote
  - conflicts (same operating signature, inconsistent outputs)

### 4) Optional LoRA/QLoRA extraction model
- Build supervision pairs: `chunk text → strict JSON records`
- Train small instruct models for consistent extraction at lower cost

---

## Architecture

```text
PDFs (data/raw_pdfs/)
   │
   ├─► Parse pages (pdf_loader) + clean text
   │
   ├─► Chunking (splitter + sectioner)
   │
   ├─► Metadata extraction
   │     ├─ rules: range/ratio/targets/stage + evidence spans
   │     └─ optional LLM augmentation (DashScope/OpenAI-compatible)
   │
   ├─► Vector DB (Chroma)  ──► Retrieval (filters + ranking) ──► Evidence-first QA
   │
   └─► Record extraction ──► SQLite (records + evidence + conflicts) ──► Export CSV/JSON
```

Storage:
- **Chroma** persistence: `storage/chroma_db/`
- **SQLite** records DB: `storage/expert.db`

---

## Quickstart

### 1) Requirements
- Python 3.10+ recommended
- macOS / Linux / WSL supported

### 2) Install

```bash
pip install -e .[dev]
pytest -q
```

If you plan to use **DashScope/Bailian** (recommended for LLM + embeddings):

```bash
pip install -U langchain-openai openai python-dotenv
```

### 3) Add PDFs
Put PDFs under:

```text
data/raw_pdfs/
```

### 4) Configure (`configs/app.yaml`)
You can use:
- **remote embeddings** via DashScope (`embeddings:` section) ✅ recommended
- **local embeddings** fallback (HF `bge-m3`) via `embedding:` section

See [Configuration guide](#configuration-guide).

### 5) Ingest into Chroma

```bash
rm -rf storage/chroma_db
mkdir -p storage/chroma_db
python scripts/ingest.py
```

Optional: enable table extraction (Camelot):

```bash
pip install camelot-py
python scripts/ingest.py --tables --table-flavor lattice
```

Optional: enable LLM metadata augmentation (**LLM is configured in app.yaml**):

```bash
python scripts/ingest.py --llm-metadata
```

### 6) Ask questions (RAG)

```bash
python scripts/ask.py
```

Example queries:
- `steam CO2 gasification NH3 HCN 1200K 2MPa`
- `在蒸汽气化条件下 NH3 和 HCN 的生成趋势？给证据`
- `热解 800C 酚类 生成机理`

Optional: LLM-grounded answer generation (still cites evidence):

```bash
python scripts/ask.py --llm
```

### 7) Extract structured records into SQLite

```bash
python scripts/extract_records.py --limit 300
```

Optional (LLM extraction):

```bash
python scripts/extract_records.py --llm --limit 300
```

### 8) Export records

```bash
python scripts/export_records.py --out data/artifacts/records.csv
```

---

## DashScope / Bailian setup

This repo supports **Alibaba Cloud Bailian (DashScope)** via **OpenAI-compatible mode**.

### 1) Set API key

Create `.env` in project root:

```env
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
```

Or export it in your shell:

```bash
export DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 2) Use the compatible base URL

In `configs/app.yaml`, use:

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

### 3) 10-second smoke test (recommended)

One-liner test for embeddings:

```bash
python -c "import os; from coal_kb.embeddings.factory import EmbeddingsConfig, make_embeddings; cfg=EmbeddingsConfig(provider='dashscope', base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', api_key_env='DASHSCOPE_API_KEY', model='text-embedding-v4', dimensions=1024); emb=make_embeddings(cfg); v=emb.embed_query('NH3 HCN formation'); print('dim=', len(v))"
```

One-liner test for chat LLM:

```bash
python -c "import os; from coal_kb.llm.factory import LLMConfig, make_chat_llm; cfg=LLMConfig(provider='dashscope', base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', api_key_env='DASHSCOPE_API_KEY', model='qwen-plus', temperature=0, timeout=60); llm=make_chat_llm(cfg); print(llm.invoke('你是谁？').content)"
```

---

## Configuration guide

### `configs/app.yaml` (overview)

Sections you will typically edit:

- `paths`: where PDFs / outputs / DBs are stored
- `chunking`: chunk size and overlap
- `chroma.collection_name`: Chroma collection name
- `embedding` (**local fallback**): e.g., `BAAI/bge-m3`
- `embeddings` (**remote**): DashScope/OpenAI-compatible embeddings
- `llm`: DashScope/OpenAI-compatible chat model

> **Important:** if you change embedding backend/model (e.g., from local to DashScope), you must **rebuild Chroma**:
>
> ```bash
> rm -rf storage/chroma_db
> python scripts/ingest.py
> ```

### Local vs remote embeddings

- `embedding:` (singular)  
  Local fallback embeddings (e.g. HuggingFace `bge-m3`). Requires `langchain-huggingface` + `sentence-transformers`.

- `embeddings:` (plural)  
  Remote embeddings (DashScope/OpenAI-compatible). Requires `langchain-openai` + key + base_url.

Recommended: start with **remote DashScope embeddings** to keep deployment simple.

---

## Pipelines & workflows

### Ingest (`scripts/ingest.py`)
Inputs: `data/raw_pdfs/*.pdf`  
Outputs:
- Chroma DB: `storage/chroma_db/`
- Enriched chunk metadata: stored as Chroma metadata

Key options:
- `--tables`: optional table extraction
- `--llm-metadata`: enable LLM metadata augmentation (reads `cfg.llm`)

### Ask (`scripts/ask.py`)
Inputs: user query  
Outputs: evidence list (and optional LLM answer)

Key options:
- `--llm`: enable LLM answer generation (reads `cfg.llm`)

### Extract records (`scripts/extract_records.py`)
Inputs: chunks (from Chroma)  
Outputs: `storage/expert.db` with structured `ExperimentRecord`s

Key options:
- `--llm`: enable LLM extraction (reads `cfg.llm`)

### Export (`scripts/export_records.py`)
Outputs: `data/artifacts/*.csv` for modeling / analysis

---

## Evidence & metadata design

Each chunk stored in Chroma carries metadata such as:
- `source_file`, `page`, `chunk_id`, `section`
- `stage`, `gas_agent`, `targets`
- `T_K`, `T_range_K`, `T_min_K`, `T_max_K`
- `P_MPa`, `P_range_MPa`, `P_min_MPa`, `P_max_MPa`
- `ratios` dict

And two audit fields:
- `meta_confidence`: `{ field: score }`
- `meta_evidence`: `{ field: {start,end,evidence_text,...} }`

LLM augmentation (if enabled) follows strict rules:
- only fill missing fields
- never overwrite numeric fields if already extracted (anti-hallucination)
- record a low-confidence marker for LLM-added fields

---

## Structured records (SQLite)

SQLite DB: `storage/expert.db`

A record typically contains:
- operating condition signature (stable hash)
- normalized condition fields (stage, gas agent, coal name, T/P, ratios)
- pollutant outputs (value + unit + basis)
- evidence quote/snippet
- conflict flags if inconsistent outputs appear under same signature

Export for training:
- `python scripts/export_records.py --out data/artifacts/records.csv`

---

## LoRA/QLoRA fine-tuning (shortest path)

### 1) Create curated pairs (JSONL)

```json
{"instruction":"Extract ExperimentRecord JSON array. Output strict JSON only. No fabrication.","input":"<chunk text>","output":"[{...}]"}
```

### 2) Train QLoRA
Install training deps:

```bash
pip install -U "trl[peft]" datasets accelerate transformers peft bitsandbytes
```

Train:

```bash
python scripts/train_lora_record_extractor.py   --model Qwen/Qwen2.5-1.5B-Instruct   --train data/artifacts/lora_train.jsonl   --val data/artifacts/lora_val.jsonl   --out data/artifacts/lora_adapter
```

Evaluate:
- JSON parse rate
- field-level precision/recall
- numeric relative error for normalized values

---

## Project structure

```text
coal-expert-kb/
  README.md
  configs/
    app.yaml
    schema.yaml
    prompts/
  data/
    raw_pdfs/
    interim/
    artifacts/
  storage/
    chroma_db/
    expert.db
  src/
    coal_kb/
      settings.py
      llm/
        factory.py
      embeddings/
        factory.py
      parsing/
      chunking/
      metadata/
      retrieval/
      qa/
      store/
      pipelines/
  scripts/
    ingest.py
    ask.py
    extract_records.py
    export_records.py
  tests/
```

---

## Testing

```bash
pytest -q
```

---

## Troubleshooting

### “My embeddings/LLM settings don’t take effect”
- Confirm `configs/app.yaml` includes `llm:` and `embeddings:` sections
- Confirm `.env` contains `DASHSCOPE_API_KEY`
- Run the smoke tests in [DashScope / Bailian setup](#dashscope--bailian-setup)

### “Chroma retrieval looks wrong after changing embedding model”
Rebuild Chroma:

```bash
rm -rf storage/chroma_db
python scripts/ingest.py
```

### LLM cost control
- Prefer rule-only ingest first (no `--llm-metadata`)
- Enable LLM augmentation only on missing-heavy chunks (default behavior)
- Consider restricting LLM augmentation to `methods/results/table` sections (roadmap)

---

## Roadmap

- Stronger section detection (Methods/Results/Table) to reduce LLM calls
- Better evidence span alignment for LLM-filled fields
- Table-first extraction (for pollutant numbers and conditions)
- Schema-constrained decoding for strict JSON extraction
- Active learning loop: conflicts → human review → curated pairs → retrain LoRA

---

## Contributing

PRs and issues are welcome:
- add ontology aliases (`configs/schema.yaml`)
- improve regex patterns (ratios/ranges)
- add unit conversions & basis normalization
- improve sectioner/table extraction
- add evaluation metrics for extraction quality

---

## License

MIT or Apache-2.0
