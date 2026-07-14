"""Normalize repository file names, references, prompts, and offline checks."""
from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()


def move(source: str, target: str) -> None:
    src, dst = ROOT / source, ROOT / target
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    src.rename(dst)


def replace(path: str, pairs: list[tuple[str, str]]) -> None:
    file_path = ROOT / path
    if not file_path.exists():
        return
    text = file_path.read_text(encoding="utf-8")
    for old, new in pairs:
        text = text.replace(old, new)
    file_path.write_text(text, encoding="utf-8")


def write(path: str, text: str, executable: bool = False) -> None:
    file_path = ROOT / path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(text, encoding="utf-8")
    if executable:
        file_path.chmod(0o755)


def process() -> None:
    renames = {
        ".github/codex/prompts/fix.md": ".github/codex/prompts/apply_fixes.md",
        ".github/codex/prompts/review.md": ".github/codex/prompts/review_pull_request.md",
        ".github/workflows/codex-fix-and-automerge.yml": ".github/workflows/codex-fix-and-auto-merge.yml",
        "configs/prompts/metadata_extract.txt": "configs/prompts/extract_metadata.txt",
        "configs/prompts/query_parse.txt": "configs/prompts/parse_query.txt",
        "configs/prompts/record_extract_optimized.txt": "configs/prompts/extract_records.txt",
        "data/eval/retrieval_gold.example.jsonl": "data/eval/retrieval_gold_sample.jsonl",
        "docs/ARCHITECTURE.md": "docs/architecture/system_overview.md",
        "docs/architecture/APPLICATION_INTERFACES.md": "docs/architecture/application_interfaces.md",
        "docs/architecture/EVALUATION_OPERATIONS.md": "docs/architecture/evaluation_operations.md",
        "docs/architecture/RETRIEVAL_ANSWERING_LAYERS.md": "docs/architecture/retrieval_answering_layers.md",
        "docs/engineering/ACCEPTANCE_CRITERIA.md": "docs/engineering/acceptance_criteria.md",
        "docs/engineering/CODING_STANDARDS.md": "docs/engineering/coding_standards.md",
        "scripts/eval.py": "scripts/evaluate_retrieval.py",
        "scripts/eval_lora_extractor.py": "scripts/evaluate_lora_record_extractor.py",
        "scripts/eval_retrieval.py": "scripts/evaluate_two_stage_retrieval.py",
        "scripts/quality/run_acceptance.sh": "scripts/quality/check_repository.sh",
        "tests/test_ask_pipeline.py": "tests/test_ask_application.py",
        "tests/test_chat_memory.py": "tests/test_conversation_history.py",
        "tests/test_constraint_policy_relax.py": "tests/test_constraint_relaxation.py",
        "tests/test_elastic_optional.py": "tests/test_elasticsearch_optional.py",
        "tests/test_loaders_text.py": "tests/test_text_loader.py",
        "tests/test_no_legacy_modules.py": "tests/test_repository_conventions.py",
        "tests/test_stage2_architecture_boundaries.py": "tests/test_architecture_boundaries.py",
    }
    for source, target in renames.items():
        move(source, target)

    old_prompt = ROOT / "configs/prompts/record_extract.txt"
    if old_prompt.exists():
        old_prompt.unlink()

    replace(".github/workflows/codex-fix-and-auto-merge.yml", [
        (".github/codex/prompts/fix.md", ".github/codex/prompts/apply_fixes.md"),
    ])
    write(".github/codex/prompts/apply_fixes.md", """You are Codex working in the Coal Expert KB repository.\n\nGoal:\nApply the requested pull-request fix with a minimal, reviewable diff while preserving the canonical architecture.\n\nRules:\n- Read `docs/engineering/coding_standards.md` and `docs/engineering/acceptance_criteria.md` first.\n- Use only the canonical packages under `src/coal_kb/`.\n- Do not create duplicate modules, alternate implementations, migration shims, or state-suffixed filenames.\n- Keep Python and Shell naming consistent with the repository conventions.\n- Update every import, configuration reference, document, test, and command affected by a rename.\n- Do not add model downloads or external-service calls to the offline test suite.\n- Never print secrets or personal data.\n\nValidation:\n- Run `bash scripts/quality/check_repository.sh`.\n- Add or update focused tests for behavior changed by the fix.\n- Report changed files, checks executed, and any check that could not run.\n""")
    write(".github/codex/prompts/review_pull_request.md", """You are Codex acting as a senior reviewer for Coal Expert KB.\n\nReview priorities:\n1. P0: security vulnerability, data loss, secret leakage, or critical outage risk.\n2. P1: high-likelihood correctness bug, breaking contract, or serious regression.\n3. P2: maintainability, naming, documentation, or minor performance improvements.\n\nRequired checks:\n- The change follows `docs/engineering/coding_standards.md`.\n- The repository keeps a single canonical implementation for each responsibility.\n- Renames update imports, configuration, scripts, tests, documentation, and Actions.\n- File names contain no migration-stage or state suffixes.\n- Offline checks remain independent of model downloads and external services.\n\nOutput format:\n## Summary\n- <1-3 bullets>\n\n## Blocking (P0/P1)\n- [P0|P1] <issue> (file: path) — <impact> — <fix>\n\n## Non-blocking (P2)\n- <optional bullets>\n\n## Validation gaps\n- <missing or insufficient checks>\n""")

    write("requirements/ci.txt", """pydantic>=2.6\npydantic-settings>=2.2\nPyYAML>=6.0\npython-dotenv>=1.0.0\nrich>=13.7\ntqdm>=4.66\nSQLAlchemy>=2.0\nfastapi>=0.115\npython-multipart>=0.0.9\nlangchain-core>=0.3\nlangchain-community>=0.3\nlangchain-chroma>=0.1.2\nlangchain-openai>=0.2\nlangchain-text-splitters>=0.3\nchromadb>=0.5\npypdf>=4.0\npymupdf>=1.24\ntiktoken>=0.7\nnumpy>=1.26\nscikit-learn>=1.4\nelasticsearch>=8.12\nuvicorn>=0.30\nrequests>=2.31\nbeautifulsoup4>=4.12\npython-docx>=1.1\npython-pptx>=0.6\nopenpyxl>=3.1\npytest>=8.0\nruff>=0.6\nmypy>=1.8\n""")
    write(".github/workflows/quality-checks.yml", """name: Quality Checks\n\non:\n  pull_request:\n  push:\n    branches: [main]\n  workflow_dispatch:\n\npermissions:\n  contents: read\n\nconcurrency:\n  group: quality-${{ github.workflow }}-${{ github.ref }}\n  cancel-in-progress: true\n\njobs:\n  offline-quality:\n    runs-on: ubuntu-latest\n    timeout-minutes: 25\n    steps:\n      - uses: actions/checkout@v5\n      - uses: actions/setup-python@v5\n        with:\n          python-version: \"3.11\"\n          cache: pip\n          cache-dependency-path: requirements/ci.txt\n      - name: Install CI dependencies\n        run: |\n          python -m pip install --upgrade pip\n          python -m pip install -r requirements/ci.txt\n          python -m pip install -e . --no-deps\n      - name: Run offline repository checks\n        env:\n          HF_HUB_OFFLINE: \"1\"\n          TRANSFORMERS_OFFLINE: \"1\"\n          TOKENIZERS_PARALLELISM: \"false\"\n        run: bash scripts/quality/check_repository.sh\n""")

    replacements = [
        ("docs/engineering/CODING_STANDARDS.md", "docs/engineering/coding_standards.md"),
        ("docs/engineering/ACCEPTANCE_CRITERIA.md", "docs/engineering/acceptance_criteria.md"),
        ("scripts/quality/run_acceptance.sh", "scripts/quality/check_repository.sh"),
        ("scripts/eval_lora_extractor.py", "scripts/evaluate_lora_record_extractor.py"),
        ("scripts/eval_retrieval.py", "scripts/evaluate_two_stage_retrieval.py"),
        ("scripts/eval.py", "scripts/evaluate_retrieval.py"),
        ("retrieval_gold.example.jsonl", "retrieval_gold_sample.jsonl"),
        ("test_no_legacy_modules.py", "test_repository_conventions.py"),
        ("test_stage2_architecture_boundaries.py", "test_architecture_boundaries.py"),
        ("test_ask_pipeline.py", "test_ask_application.py"),
        ("test_chat_memory.py", "test_conversation_history.py"),
        ("test_constraint_policy_relax.py", "test_constraint_relaxation.py"),
        ("test_elastic_optional.py", "test_elasticsearch_optional.py"),
        ("test_loaders_text.py", "test_text_loader.py"),
        ("configs/prompts/record_extract.txt", "configs/prompts/extract_records.txt"),
    ]
    for path in ["README.md", "scripts/quality/config.sh", "docs/engineering/acceptance_criteria.md", "docs/engineering/coding_standards.md", "src/coal_kb/records/pipeline.py"]:
        replace(path, replacements)

    replace("configs/app.yaml", [("# legacy fallback", "# section-aware fallback")])
    replace("src/coal_kb/infra/config/models.py", [("# legacy fallback options", "# section-aware fallback options")])
    replace("src/coal_kb/ingestion/chunking/splitter.py", [("Legacy character splitter.", "Section-aware character splitter.")])
    replace("src/coal_kb/ingestion/pipeline.py", [("strategy == \"legacy\"", "strategy == \"section_aware\"")])
    replace("tests/test_architecture_boundaries.py", [("第二阶段", "核心")])
    replace("tests/test_pdf_markdown_quality.py", [("from coal_kb.ingestion.loaders.markdown_clean import collapse_repeated_headers, fix_hyphenation\n\nfrom coal_kb.infra.config import PDFMarkdownConfig\nfrom coal_kb.ingestion.loaders.pdf_loader import PDFLoader", "from coal_kb.infra.config import PDFMarkdownConfig\nfrom coal_kb.ingestion.loaders.markdown_clean import collapse_repeated_headers, fix_hyphenation\nfrom coal_kb.ingestion.loaders.pdf_loader import PDFLoader")])

    replace("scripts/evaluate_retrieval.py", [
        ("class Eval:", "class EvaluateRetrieval:"),
        ("Eval(\n", "EvaluateRetrieval(\n"),
        ("python scripts/eval.py", "python scripts/evaluate_retrieval.py"),
    ])
    replace("scripts/evaluate_two_stage_retrieval.py", [
        ("class EvalRetrieval:", "class EvaluateTwoStageRetrieval:"),
        ("EvalRetrieval(\n", "EvaluateTwoStageRetrieval(\n"),
        ("python scripts/eval_retrieval.py", "python scripts/evaluate_two_stage_retrieval.py"),
    ])
    replace("scripts/evaluate_lora_record_extractor.py", [
        ("python scripts/eval_lora_extractor.py", "python scripts/evaluate_lora_record_extractor.py"),
    ])

    replace("docs/engineering/coding_standards.md", [("- 合并前执行", "## 文件命名\n\n- Python、Shell、配置、数据和文档文件使用小写 `snake_case`。\n- GitHub Actions 工作流使用小写 `kebab-case.yml`。\n- `README.md`、`LICENSE`、`Dockerfile`、`pyproject.toml` 等行业标准名称可以保留。\n- 文件名不得包含迁移阶段号或状态后缀。\n- 样例数据使用 `_sample` 后缀。\n\n- 合并前执行")])
    replace("docs/engineering/acceptance_criteria.md", [
        ("本阶段不建立 GitHub Actions，也不执行可能调用外部模型或消耗 Token 的全量测试。合并依据是可重复的本地离线验收。", "GitHub Actions 与本地检查共用同一套离线验收命令，不访问外部 LLM、Embedding API 或真实 Elasticsearch。"),
        ("不执行全量 `pytest`，不以未运行的测试作为通过依据。", "需要 Token 或真实索引的端到端评估单独执行。"),
    ])

    conventions = ROOT / "tests/test_repository_conventions.py"
    text = conventions.read_text(encoding="utf-8")
    text = text.replace("确保已删除的模块、备份文件和 import 不会重新进入仓库。", "确保仓库只保留正式结构、规范文件名和有效 import。")
    text = text.replace("REMOVED_PATHS", "DISALLOWED_PATHS").replace("REMOVED_MODULES", "DISALLOWED_MODULES").replace("_is_removed_module", "_is_disallowed_module")
    text = text.replace("test_removed_paths_do_not_exist", "test_disallowed_paths_do_not_exist")
    text = text.replace("test_backup_sources_are_absent", "test_generated_and_backup_sources_are_absent")
    text = text.replace("for removed in DISALLOWED_MODULES", "for disallowed in DISALLOWED_MODULES").replace("removed or", "disallowed or").replace("f\"{removed}.\"", "f\"{disallowed}.\"")
    text += '''\n\ndef test_repository_file_names_are_normalized() -> None:\n    import re\n\n    standard = {\"README.md\", \"LICENSE\", \"Dockerfile\", \"pyproject.toml\", \"docker-compose.yml\", \"__init__.py\", \"config.py\", \"config.sh\", \"conftest.py\", \"index.html\", \"app.js\", \"styles.css\"}\n    snake = re.compile(r\"^[a-z0-9]+(?:_[a-z0-9]+)*$\")\n    kebab = re.compile(r\"^[a-z0-9]+(?:-[a-z0-9]+)*$\")\n    state = re.compile(r\"(?:^|_)(?:old|backup|bak|final|optimized|copy|stage\\d+)(?:_|\\.|$)\")\n    violations: list[str] = []\n    for path in REPO_ROOT.rglob(\"*\"):\n        if not path.is_file() or {\".git\", \"__pycache__\", \".pytest_cache\", \".mypy_cache\", \".ruff_cache\"}.intersection(path.parts):\n            continue\n        name = path.name\n        if name in standard or name.startswith(\".\"):\n            continue\n        stem = path.stem\n        if state.search(name.lower()):\n            violations.append(str(path.relative_to(REPO_ROOT)))\n        elif path.relative_to(REPO_ROOT).parts[:2] == (\".github\", \"workflows\"):\n            if path.suffix not in {\".yml\", \".yaml\"} or not kebab.fullmatch(stem):\n                violations.append(str(path.relative_to(REPO_ROOT)))\n        elif path.suffix in {\".py\", \".sh\", \".md\", \".txt\", \".jsonl\", \".yaml\", \".yml\"} and not snake.fullmatch(stem):\n            violations.append(str(path.relative_to(REPO_ROOT)))\n    assert violations == []\n'''
    conventions.write_text(text, encoding="utf-8")

    for path in ROOT.rglob("*"):
        if not path.is_file() or {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}.intersection(path.parts):
            continue
        if path.suffix.lower() not in {".py", ".sh", ".md", ".txt", ".yaml", ".yml", ".jsonl"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = text.replace("legacy", "previous").replace("Legacy", "Previous")
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    process()
