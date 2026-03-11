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
- FastAPI + static three-panel chat frontend for interactive demos
- Persistent conversation history with evidence-side inspection
- Local browser settings for runtime request overrides

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
  -> conversation memory / follow-up handling
  -> QueryPlanner
  -> ExpertRetriever
  -> ContextBuilder
  -> Answerer
  -> citation catalog + diagnostics
  -> chat UI / CLI rendering
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

The repository now ships with a conversation-first frontend built as a lightweight static app served by FastAPI.

### Why this frontend approach

- It keeps the stack simple enough to study end-to-end.
- It still feels like a credible mini product instead of a toy demo.
- It makes evidence inspection a first-class interaction instead of a hidden debug view.

### UI layout

```text
| Conversations | Chat Thread + Composer | Evidence / Source / Diagnostics |
```

The app is desktop-first and organized into three working areas:

- Left sidebar
  - conversation history
  - new chat action
  - quick runtime settings summary
- Center pane
  - message thread
  - grounded assistant answers
  - inline citation chips
  - message composer
- Right inspector
  - citation references
  - evidence cards
  - source cards
  - retrieval diagnostics

### Frontend features

- persistent conversation list backed by the API
- assistant message selection for evidence inspection
- loading, empty, and error states
- inline citation-aware answer rendering
- claim map for assistant turns
- source cards with file, page, heading, and preview text
- settings drawer with local browser persistence
- dark/light presentation toggle

### Settings drawer

The settings drawer is intentionally practical and learning-oriented. It lets you configure:

- API base URL
- provider base URL
- API key
- LLM provider
- LLM model
- embedding model
- retrieval backend
- retrieval mode
- top-k
- rerank on/off
- LLM answer generation on/off
- debug mode on/off

These settings are stored in `localStorage` and sent with each chat request where applicable.

Important note:

- changing the embedding model at query time only makes sense if it matches the embedding space used to build the index

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

### Frontend-facing settings endpoint

The frontend loads backend defaults from:

- `GET /api/settings/defaults`

This returns the current server defaults plus supported option lists so the settings drawer can initialize cleanly.

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
    runtime_overrides.py      Request-time model / provider overrides for the UI
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
      index.html              Three-pane chat app shell
      styles.css              Frontend layout, theme, and visual system
      app.js                  Chat state, API integration, and settings persistence

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

The frontend is served by the same FastAPI process as the API.

### Start the backend + frontend together

```bash
python scripts/serve.py --reload
```

Then open:

```text
http://127.0.0.1:8000
```

### First-run checklist

1. Start the API with `python scripts/serve.py --reload`
2. Open the UI in your browser
3. Open the settings drawer
4. Confirm the API base URL if you are not using the same origin
5. Add an API key and provider base URL if your embedding / LLM providers require overrides
6. Confirm retrieval backend, mode, and top-k
7. Start a new conversation and ask a grounded question

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

The primary UI flow is conversation-based:

- `GET /api/settings/defaults`
- `GET /api/conversations`
- `GET /api/conversations/{conversation_id}/messages`
- `POST /api/chat`
- `DELETE /api/conversations/{conversation_id}`

`POST /api/ask` still exists for single-turn workflows and CLI parity, but the web app now centers on `POST /api/chat`.

### Chat endpoint

Request:

```json
{
  "conversation_id": null,
  "message": "How does steam gasification affect NH3 and HCN at 1200 K?",
  "llm": true,
  "debug": true,
  "api_key": "optional-request-time-key",
  "provider_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "llm_model": "qwen-plus",
  "embedding_model": "text-embedding-v3",
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
- `source_cards`
- `rendered_citations`
- `retrieval_trace_summary`
- `evidence_sufficiency`
- `confidence_score`

### How the UI uses chat responses

For each assistant message, the frontend stores and renders:

- `answer` as the visible assistant message body
- `rendered_citations` as compact inline evidence chips
- `claim_items` as a claim-to-evidence map in the inspector
- `citations` as evidence cards in the inspector
- `source_cards` as grouped source summaries
- `diagnostics` as developer-facing retrieval/context details
- `evidence_sufficiency` and `confidence_score` as quick trust signals

## Example workflows

### Workflow 1: ingest and ask from the CLI

```bash
python scripts/index.py build --embedding-version v1
python scripts/ask.py --backend elastic --mode balanced
```

### Workflow 2: use the conversation UI

```bash
python scripts/serve.py --reload
```

Then:

1. Open the settings drawer and confirm your runtime configuration.
2. Start a new chat or open an existing conversation from the left sidebar.
3. Enter a question in the center composer.
4. Read the grounded assistant answer in the thread view.
5. Click the assistant message to inspect its evidence panel on the right.
6. Review citation references, evidence cards, source cards, and diagnostics.
7. Ask a follow-up question in the same thread to exercise conversation-aware retrieval.

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
- Runtime settings overrides are applied request-by-request instead of introducing a heavy admin/config subsystem.

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
