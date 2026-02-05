# Coal Expert KB (Modern RAG)

Modernized Elastic-first RAG system for coal pyrolysis/gasification knowledge.

## Quick start

```bash
pip install -e .[dev]
PYTHONPATH=src python scripts/index.py build
PYTHONPATH=src python scripts/ask.py
```

## Main commands

- Ingest: `PYTHONPATH=src python scripts/ingest.py`
- Index build/switch/rollback: `PYTHONPATH=src python scripts/index.py ...`
- Ask: `PYTHONPATH=src python scripts/ask.py`
- Eval: `PYTHONPATH=src python scripts/eval.py --task retrieval`

## Architecture

See `docs/ARCHITECTURE.md` for layers and migration notes.

## Key capabilities

- PDF markdown-first loading (dict/layout-aware) with text fallback.
- Hierarchical parent/child chunking with semantic boundaries.
- Elastic parent→child two-stage retrieval with fallback baseline.
- ContextBuilder with token budget + citation ids.
- Citation-grounded answerer with uncertainty mode.

## Migration notes

This version changes Elastic mapping and requires index rebuild:

```bash
PYTHONPATH=src python scripts/index.py build
```

Rollback toggles:
- `retrieval.two_stage.enabled: false`
- `chunking.strategy: legacy`
- `pdf_markdown.enabled: false`
