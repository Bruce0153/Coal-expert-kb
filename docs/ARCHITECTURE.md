# Coal KB Modern RAG Architecture

## Layered modules

- `coal_kb.ingestion`: scanning/loading/chunking-oriented stage primitives.
- `coal_kb.indexing`: index build/validation helpers.
- `coal_kb.query`: `QueryPlanner` + `QueryPlan` planning layer.
- `coal_kb.retrieval`: plan executor (`ExpertRetriever.execute(plan)`) with Elastic two-stage parent→child retrieval.
- `coal_kb.context`: `ContextBuilder` for evidence organization, token budgeting and citation ids.
- `coal_kb.generation`: citation-grounded `Answerer` with uncertainty fallback and post-check.
- `coal_kb.evaluation`: unified `EvalRunner`.
- `coal_kb.observability`: query log schema.
- `coal_kb.knowledge_extraction`: extraction plugin APIs.

## Data flow

1. User query -> `QueryPlanner.build_plan`.
2. `ExpertRetriever.execute(plan)` executes Stage-1 parents then Stage-2 children.
3. `ContextBuilder` builds structured prompt context and `[Sx]` mapping.
4. `Answerer` produces grounded answer with uncertainty/rejection behavior.
5. Query execution metadata is logged into registry.

## Retrieval strategy

- Stage-1: `is_parent=true`, strict hard constraints.
- Stage-2: `is_parent=false` + `terms(parent_id in stage1_ids)`.
- fallback: if stage1 empty or stage2 empty -> baseline child retrieval.
- optional neighbor expansion around top child positions.

## Elastic mapping migration

Mapping includes parent/child fields:
`is_parent`, `parent_id`, `heading_path`, `heading_path_text`, `chunk_level`, `position_start`, `position_end`.

Rebuild required after migration:

```bash
PYTHONPATH=src python scripts/index.py build
```

## Cleanup

### Moved / frozen
- Legacy chunking implementation moved to `src/coal_kb/chunking/legacy/advanced_splitter.py`; old import path kept as shim for rollback compatibility.
- LoRA training/evaluation scripts moved to `experiments/` and kept wrappers in `scripts/`.

### Removed stale artifacts
- Removed committed packaging artifacts under `src/coal_expert_kb.egg-info/`.
- Removed backup files: `src/coal_kb/settings.py.bak_ingestion_cfg`, `src/coal_kb/retrieval/retriever.py.bak_soft_rank`.

### Fallback switches
- `chunking.strategy: legacy` keeps old split behavior path.
- `retrieval.two_stage.enabled: false` forces baseline single-stage retrieval.
- `pdf_markdown.enabled: false` disables markdown-first PDF extraction.
