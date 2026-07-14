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
  "$REPO_ROOT/src/coal_kb/evaluation"
  "$REPO_ROOT/src/coal_kb/application"
  "$REPO_ROOT/src/coal_kb/interfaces/api"
)
ARCHITECTURE_TESTS=(
  "$REPO_ROOT/tests/test_repository_conventions.py"
  "$REPO_ROOT/tests/test_deprecated_markers.py"
  "$REPO_ROOT/tests/test_architecture_boundaries.py"
  "$REPO_ROOT/tests/test_evaluation_operations_architecture.py"
)
FOUNDATION_TESTS=(
  "$REPO_ROOT/tests/test_text_clean.py"
  "$REPO_ROOT/tests/test_units.py"
  "$REPO_ROOT/tests/test_validators.py"
  "$REPO_ROOT/tests/test_conversation_store.py"
  "$REPO_ROOT/tests/test_registry.py"
  "$REPO_ROOT/tests/test_config_consistency.py"
  "$REPO_ROOT/tests/test_evaluation_pipeline.py"
)
RAG_TESTS=(
  "$REPO_ROOT/tests/test_context_builder.py"
)
