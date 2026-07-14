# 本地离线验收使用的路径和测试清单。

REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
RUFF_BIN=${RUFF_BIN:-ruff}
MYPY_BIN=${MYPY_BIN:-mypy}
PYTEST_BIN=${PYTEST_BIN:-pytest}
TEST_TIMEOUT_SECONDS=${TEST_TIMEOUT_SECONDS:-180}

SOURCE_PATHS=("$REPO_ROOT/src/coal_kb" "$REPO_ROOT/scripts" "$REPO_ROOT/tests")
MYPY_TARGETS=(
  "$REPO_ROOT/src/coal_kb/core"
  "$REPO_ROOT/src/coal_kb/infra/config"
  "$REPO_ROOT/src/coal_kb/retrieval"
  "$REPO_ROOT/src/coal_kb/context"
  "$REPO_ROOT/src/coal_kb/answering"
  "$REPO_ROOT/src/coal_kb/application"
  "$REPO_ROOT/src/coal_kb/interfaces/api"
)
ARCHITECTURE_TESTS=(
  "$REPO_ROOT/tests/test_no_legacy_modules.py"
  "$REPO_ROOT/tests/test_stage2_architecture_boundaries.py"
  "$REPO_ROOT/tests/test_evaluation_operations_architecture.py"
)
FOUNDATION_TESTS=(
  "$REPO_ROOT/tests/test_text_clean.py"
  "$REPO_ROOT/tests/test_units.py"
  "$REPO_ROOT/tests/test_validators.py"
  "$REPO_ROOT/tests/test_conversation_store.py"
  "$REPO_ROOT/tests/test_registry.py"
)
RAG_TESTS=(
  "$REPO_ROOT/tests/test_context_builder.py"
  "$REPO_ROOT/tests/test_retrieval.py"
  "$REPO_ROOT/tests/test_retrieval_diversity.py"
  "$REPO_ROOT/tests/test_retrieval_reference_filter.py"
  "$REPO_ROOT/tests/test_soft_scoring_missing_metadata.py"
  "$REPO_ROOT/tests/test_loaders_text.py"
  "$REPO_ROOT/tests/test_loader_registry.py"
  "$REPO_ROOT/tests/test_markdown_hierarchical_semantic.py"
  "$REPO_ROOT/tests/test_pdf_loader_markdown_fallback.py"
  "$REPO_ROOT/tests/test_pdf_markdown_quality.py"
  "$REPO_ROOT/tests/test_table_extractor.py"
  "$REPO_ROOT/tests/test_query_planner.py"
  "$REPO_ROOT/tests/test_ask_pipeline.py"
  "$REPO_ROOT/tests/test_chat_memory.py"
  "$REPO_ROOT/tests/test_chat_orchestrator.py"
  "$REPO_ROOT/tests/test_api_runtime_overrides.py"
)
