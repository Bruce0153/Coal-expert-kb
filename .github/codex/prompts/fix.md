You are Codex. You have access to this repository workspace.

Goal:
Implement 3 product improvements (minimal, safe diffs), keeping existing tests passing.

Rules:
- Follow AGENTS.md.
- Keep diffs minimal; avoid unrelated refactors.
- Do NOT add heavy/new dependencies unless truly necessary.
- Add minimal tests for any changed behavior; tests must not download large models.
- Never print secrets/PII.

Tasks (do all):
1) Wire in optional CrossEncoder reranker to retrieval
   - There is already CrossEncoderReranker in src/coal_kb/retrieval/rerank.py.
   - Integrate it into ExpertRetriever.retrieve() in src/coal_kb/retrieval/retriever.py:
     After RRF fuse and BEFORE post-filter (or after post-filter; choose the safer one), apply reranker on top-N candidates.
   - Make it OPTIONAL and OFF by default.
   - Add config support:
     - Add a retrieval config section (or extend existing config) in configs/app.yaml schema + src/coal_kb/settings models so user can set:
       - retrieval.rerank_enabled: bool (default false)
       - retrieval.rerank_model: str (default cross-encoder/ms-marco-MiniLM-L-6-v2)
       - retrieval.rerank_top_k: int (default 20)
   - Update scripts/ask.py to expose CLI flags:
     --rerank (enable), --rerank-model, --rerank-top-k
   - Add a minimal unit test that verifies when rerank is disabled the ordering is unchanged,
     and when rerank enabled but sentence-transformers is missing, it falls back gracefully (no crash).

2) Add retrieval evaluation script + template dataset
   - Add scripts/eval_retrieval.py that:
     - loads a jsonl file (default data/eval/retrieval_gold.jsonl)
     - for each sample, runs FilterParser + ExpertRetriever and computes:
       Recall@K (K=1,3,5), and filter correctness (stage/gas/targets/T/P if present)
     - prints a small summary table to stdout
   - Add data/eval/retrieval_gold.example.jsonl with ~3 example rows users can copy and extend.
   - Add README snippet on how to run:
     python scripts/eval_retrieval.py --gold data/eval/retrieval_gold.jsonl --k 5

3) Table extraction auto strategy (lattice -> stream fallback)
   - Extend TableExtractor in src/coal_kb/parsing/table_extractor.py to accept flavor="auto".
   - Behavior: try lattice first; if it returns 0 tables or throws, fallback to stream.
   - Update scripts/ingest.py to allow --table-flavor auto (choices: lattice, stream, auto).
   - Ensure existing behavior unchanged when flavor is explicitly lattice/stream.
   - Add a small unit test for TableExtractor.auto that mocks camelot.read_pdf calls:
     first lattice returns empty -> stream called -> returns tables -> output docs non-empty.

Deliverable:
- Make changes directly in repo.
- Ensure pytest passes.
- Provide a short summary of changes.
