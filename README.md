# Coal Expert KB

An evidence-grounded RAG system for coal pyrolysis and gasification literature.

This repository is designed as both:

- a serious RAG learning project
- a portfolio-quality demo of auditable retrieval, cite-aware answering, and lightweight product UX

It is not just "chat with PDFs." The system parses scientific documents, builds deterministic metadata, retrieves evidence with traceability, and answers with explicit citation labels that map back to source snippets.

## Why this project exists

Scientific and engineering questions are rarely satisfied by a generic chatbot answer. In this domain, users often need to know:

- which paper supports a claim
- which section or page contains the evidence
- whether the answer is actually grounded or only plausible
- what operating conditions were mentioned in the supporting text

Coal Expert KB is built to make those questions inspectable.

## What makes it different from a basic RAG demo

- Retrieval is auditable rather than opaque.
- Metadata is deterministic rather than LLM-generated.
- Answer generation is evidence-first and cite-aware.
- The frontend makes the answer and the supporting evidence visible side by side.
- The repository is organized so a new developer can study ingestion, chunking, retrieval, context construction, answer generation, and UI in one place.

## Core capabilities

- Multi-format ingestion for PDFs and document files
- Hierarchical / section-aware chunking
- Deterministic metadata extraction for retrieval and traceability
- Chroma or Elasticsearch retrieval backends
- Query planning and constraint-aware retrieval
- Cite-aware answer generation with evidence labels like `[E1]`
- CLI experience for debugging retrieval behavior
- FastAPI + static frontend for interactive demos

## Architecture

### End-to-end flow

```text
Raw documents
  -> loaders / parsing
  -> cleaning and page extraction
  -> chunking
  -> deterministic metadata extraction
  -> vector / search indexing
  -> query planning
  -> retrieval + optional reranking
  -> context building with evidence labels
  -> answer generation with inline citations
  -> CLI or web UI
```

### Main runtime path

```text
User question
  -> QueryPlanner
  -> ExpertRetriever
  -> ContextBuilder
  -> Answerer
  -> citation catalog + diagnostics
```

## Cite-aware answering

The cite-aware path is one of the main teaching goals of this project.

### How grounding works

1. Retrieved chunks are deduplicated and packed into a bounded evidence catalog.
2. Each evidence chunk receives a stable label such as `[E1]`, `[E2]`, and so on.
3. The answer prompt instructs the model to cite those labels inline for every material claim.
4. If the model does not use valid evidence labels, the system falls back to an evidence-only answer.
5. The UI and CLI render the answer separately from the evidence catalog so the user can verify claims quickly.

### What appears in each citation

Each citation includes:

- evidence label
- source file name
- page number when available
- heading / section when available
- snippet text
- chunk id in the API payload for debugging

### Why metadata is deterministic

This project intentionally avoids LLM-generated metadata enrichment.

Metadata is limited to things that are stable and explainable, such as:

- source file path
- document id / chunk id
- title
- page / heading / section
- language
- token and character counts
- rule-based operating-condition fields such as stage, targets, and temperature / pressure ranges

That keeps retrieval filters honest and makes debugging much easier.

### Current limitations

- Answer quality still depends on retrieval quality.
- Citation labels map to retrieved evidence chunks, not sentence-level spans.
- If the retrieved evidence is weak, the answer can only say that evidence is insufficient.

## Frontend

The repository now includes a lightweight web app built with:

- FastAPI for the backend endpoint
- static HTML, CSS, and JavaScript for the frontend

This choice keeps the project easy to study and avoids unnecessary framework sprawl.

### Frontend goals

- present answer and evidence together
- make the evidence-first nature of the system visually obvious
- provide a polished demo without hiding the engineering

### Frontend features

- query input and ask action
- loading, empty, and error states
- answer panel with inline evidence labels
- evidence cards showing file, page, heading, and snippet
- diagnostics panel for retrieval and context details
- theme toggle for presentation polish

## Conversation-capable API

The backend now supports conversation-oriented chat sessions instead of only single-turn asks.

### Conversation model

- A conversation has a `conversation_id`, title, timestamps, and ordered messages.
- Each message has a `message_id`, `role`, `content`, and optional metadata.
- Roles are `system`, `user`, and `assistant`.
- Conversations are persisted in SQLite so a future chat UI can load prior threads.

### Multi-turn behavior

The backend does not blindly stuff full history into retrieval.

Instead it:

1. stores the full transcript
2. looks at a small recent window for follow-up detection
3. rewrites the retrieval query only when the new turn looks referential
4. passes a compact history view to answer generation for continuity

This keeps retrieval explainable while still supporting follow-up questions like:

- "What about CO2 instead?"
- "Compare that with steam gasification."
- "And under higher pressure?"

## Repository structure

```text
configs/
  app.yaml                    Main configuration
  schema.yaml                 Ontology and normalization rules

data/
  raw_pdfs/                   Source PDFs
  raw_docs/                   Source non-PDF docs
  interim/                    Cached intermediate outputs

scripts/
  ingest.py                   Incremental ingestion
  index.py                    Build / validate index workflow
  ask.py                      CLI ask entrypoint
  serve.py                    Web server entrypoint

src/coal_kb/
  api/
    app.py                    FastAPI app and /api/ask endpoint
    models.py                 API request / response models
    routes_chat.py            Conversation-oriented chat routes
  chat/
    orchestrator.py           Multi-turn chat orchestration
    memory.py                 Lightweight history-aware query preparation
  conversation/
    store.py                  SQLite-backed conversation persistence
    service.py                Conversation lifecycle helpers
    models.py                 Conversation and message schemas
  chunking/                   Chunking logic
  context/
    builder.py                Evidence packing and citation labeling
    types.py                  Context and citation schemas
  generation/
    answerer.py               Answer synthesis and grounding checks
  metadata/
    extract.py                Deterministic metadata extraction
  pipelines/
    ingest_pipeline.py        Ingestion and indexing pipeline
  qa/
    ask_pipeline.py           Shared ask pipeline for CLI and API
  retrieval/                  Retrieval, reranking, and filtering
  store/                      Storage integrations
  web/
    static/
      index.html              Frontend shell
      styles.css              UI styling
      app.js                  Frontend behavior

tests/
  test_context_builder.py
  test_ask_integration.py
  test_ask_pipeline.py
  ...
```

## Setup

### Requirements

- Python 3.10+
- pip
- Docker and Docker Compose if you want local Elasticsearch

### Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e .[dev,docs]
```

If you are on Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .[dev,docs]
```

### Environment variables

Create a `.env` file in the repo root when using DashScope / OpenAI-compatible services:

```env
DASHSCOPE_API_KEY=your_api_key_here
```

### Configure the app

The main settings file is:

- `configs/app.yaml`

Key things you may want to change:

- retrieval backend
- embedding model
- reranker settings
- Elasticsearch host
- paths for raw docs and storage

## Build the knowledge base

### Add documents

Put source documents in:

```text
data/raw_pdfs/
data/raw_docs/
```

### Start Elasticsearch

```bash
docker compose up -d
```

### Build an index

For a clean build with validation:

```bash
python scripts/index.py build --embedding-version v1
```

For incremental ingestion:

```bash
python scripts/ingest.py
```

## Ask questions in the CLI

Interactive mode:

```bash
python scripts/ask.py --backend elastic --mode balanced
```

One-shot mode:

```bash
python scripts/ask.py --backend elastic --mode balanced "How does steam gasification influence NH3 and HCN formation near 1200 K?"
```

Enable LLM answer generation:

```bash
python scripts/ask.py --backend elastic --mode balanced --llm
```

Enable debug output:

```bash
python scripts/ask.py --backend elastic --debug
```

### CLI commands

Inside interactive mode:

- `help`
- `debug`
- `exit`
- `quit`

## Run the frontend

Start the server:

```bash
python scripts/serve.py --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## Run the API for chat sessions

Start the same server:

```bash
python scripts/serve.py --reload
```

The key endpoints for the next frontend phase are:

- `POST /api/chat`
- `POST /api/conversations`
- `GET /api/conversations`
- `GET /api/conversations/{conversation_id}/messages`
- `DELETE /api/conversations/{conversation_id}`

### What the frontend calls

Endpoint:

```text
POST /api/ask
```

Example request body:

```json
{
  "query": "How does steam gasification influence NH3 and HCN formation near 1200 K?",
  "llm": true,
  "debug": true,
  "k": 6,
  "mode": "balanced",
  "rerank": true
}
```

Example response shape:

```json
{
  "query": "...",
  "answer": "## Answer\n...\n[E1]",
  "referenced_labels": ["E1", "E2"],
  "citations": [
    {
      "label": "E1",
      "source_file": "paper-a.pdf",
      "page": 4,
      "heading_path": "Results",
      "chunk_id": "abc123",
      "snippet": "...",
      "source_display": "paper-a.pdf | page 4 | Results",
      "referenced_in_answer": true
    }
  ],
  "timings_ms": {
    "plan": 1.2,
    "retrieve": 14.5,
    "context": 0.8,
    "answer": 420.1,
    "total": 436.6
  },
  "diagnostics": {
    "retrieval": [],
    "context": {}
  }
}
```

### Chat endpoint

Request:

```json
{
  "conversation_id": null,
  "message": "How does steam gasification affect NH3 and HCN at 1200 K?",
  "llm": true,
  "debug": true,
  "k": 6,
  "mode": "balanced",
  "rerank": true
}
```

Response fields now include:

- `conversation_id`
- `message_id`
- `answer`
- `citations`
- `used_chunks`
- `evidence_items`
- `retrieval_trace_summary`
- `evidence_sufficiency`
- `confidence_score`

## Example workflows

### Workflow 1: ingest and ask from the CLI

```bash
python scripts/index.py build --embedding-version v1
python scripts/ask.py --backend elastic --mode balanced
```

### Workflow 2: use the frontend demo

```bash
python scripts/serve.py --reload
```

Then:

1. Enter a question in the web UI.
2. Review the answer section.
3. Inspect inline labels such as `[E1]`.
4. Read the matching evidence cards on the right.
5. Open diagnostics if you want retrieval visibility.

### Workflow 3: multi-turn chat via API

1. `POST /api/conversations` to create a thread, or let `POST /api/chat` create one implicitly.
2. Send the first user message to `POST /api/chat`.
3. Reuse the returned `conversation_id` for follow-up turns.
4. Load message history with `GET /api/conversations/{conversation_id}/messages`.

### Workflow 4: evidence-only behavior

If you keep LLM answering disabled, the system still returns:

- the evidence catalog
- evidence sufficiency messaging
- stable evidence labels for auditing

This is useful for debugging retrieval independently of generation.

## Educational design choices

This repository intentionally favors clarity over excessive abstraction.

Examples:

- The frontend is plain HTML/CSS/JS so developers can inspect the full stack quickly.
- The ask pipeline is shared between CLI and API to show how product surfaces can reuse the same RAG core.
- Metadata extraction is deterministic so retrieval behavior can be reasoned about.

## Testing

Run the tests:

```bash
pytest -q
```

If you only want to validate syntax quickly:

```bash
python -m compileall src scripts tests
```

## Roadmap

Good next steps for future contributors:

- hybrid retrieval visualization in the frontend
- richer metadata filters in the web UI
- streaming answers
- conversation history
- stronger citation validation against answer sentences
- better evaluation datasets for retrieval and faithfulness

## Honest status

This is already a strong learning and demo project, but it is still a project for studying and extension rather than a hardened production system.

The current strengths are:

- retrieval traceability
- evidence-first answer packaging
- practical end-to-end developer experience
- a frontend that demonstrates how grounded RAG can feel like a product

The main remaining gaps are:

- deeper automated evaluation of citation quality
- more advanced hybrid retrieval
- more robust production deployment ergonomics

## License

Apache-2.0
