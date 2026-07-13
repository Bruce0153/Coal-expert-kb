"""验证 ingestion/tokenization 新路径与旧兼容路径行为一致。"""

from __future__ import annotations

import ast
from pathlib import Path

from coal_kb.chunking.tokenizer import count_tokens as legacy_count_tokens
from coal_kb.ingestion.chunking.sectioner import is_reference_like as canonical_reference_check
from coal_kb.ingestion.loaders.pdf_loader import PDFLoader as CanonicalPDFLoader
from coal_kb.ingestion.pipeline import IngestPipeline as CanonicalIngestPipeline
from coal_kb.loaders.pdf_loader import PDFLoader as LegacyPDFLoader
from coal_kb.pipelines.ingest_pipeline import IngestPipeline as LegacyIngestPipeline
from coal_kb.tokenization import count_tokens as canonical_count_tokens


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_legacy_ingestion_exports_point_to_canonical_objects() -> None:
    assert LegacyPDFLoader is CanonicalPDFLoader
    assert LegacyIngestPipeline is CanonicalIngestPipeline
    assert legacy_count_tokens is canonical_count_tokens
    assert canonical_reference_check(
        "References\n"
        "1. Smith A. Fuel 2020\n"
        "2. Jones B. Fuel 2021\n"
        "3. Lee C. Energy & Fuels 2022\n"
        "4. Wang D. Combustion 2023"
    )


def test_ingest_pipeline_exposes_process_and_legacy_run() -> None:
    assert callable(CanonicalIngestPipeline.process)
    assert callable(CanonicalIngestPipeline.run)


def test_active_internal_code_uses_canonical_ingestion_imports() -> None:
    root = Path("src/coal_kb")
    compatibility_roots = {
        root / "loaders",
        root / "parsing",
        root / "metadata",
        root / "chunking",
    }
    compatibility_files = {
        root / "pipelines/ingest_pipeline.py",
        root / "utils/text_clean.py",
    }
    forbidden = (
        "coal_kb.loaders",
        "coal_kb.parsing",
        "coal_kb.metadata",
        "coal_kb.chunking",
        "coal_kb.pipelines.ingest_pipeline",
        "coal_kb.utils.text_clean",
    )
    violations: list[str] = []
    for path in root.rglob("*.py"):
        if path in compatibility_files:
            continue
        if any(path.is_relative_to(directory) for directory in compatibility_roots):
            continue
        for module in _imports(path):
            if module.startswith(forbidden):
                violations.append(f"{path}: {module}")
    assert not violations, "Legacy ingestion imports remain:\n" + "\n".join(violations)
