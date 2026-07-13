"""验证可执行 Python 脚本遵循项目约定。"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SCRIPT_CLASSES = {
    "ask.py": "Ask",
    "build_lora_dataset.py": "BuildLoraDataset",
    "eval.py": "Eval",
    "eval_lora_extractor.py": "EvalLoraExtractor",
    "eval_retrieval.py": "EvalRetrieval",
    "export_records.py": "ExportRecords",
    "extract_records.py": "ExtractRecords",
    "index.py": "Index",
    "ingest.py": "Ingest",
    "serve.py": "Serve",
    "train_lora_record_extractor.py": "TrainLoraRecordExtractor",
    "validate_index.py": "ValidateIndex",
}


def _read_script(name: str) -> tuple[str, ast.Module]:
    text = (Path("scripts") / name).read_text(encoding="utf-8")
    return text, ast.parse(text)


def test_scripts_use_single_line_docstring_and_run_comment() -> None:
    for name in SCRIPT_CLASSES:
        text, tree = _read_script(name)
        assert tree.body, name
        first = tree.body[0]
        assert isinstance(first, ast.Expr), name
        assert isinstance(first.value, ast.Constant) and isinstance(first.value.value, str), name
        assert "\n" not in first.value.value, name
        last_line = next(line for line in reversed(text.splitlines()) if line.strip())
        assert last_line.startswith("# 运行命令："), name


def test_scripts_define_matching_stateful_step_with_process() -> None:
    for name, expected_class in SCRIPT_CLASSES.items():
        _, tree = _read_script(name)
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        assert expected_class in classes, name
        methods = {
            node.name
            for node in classes[expected_class].body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert "process" in methods, name


def test_random_sampling_uses_module_seed_without_global_state() -> None:
    text, _ = _read_script("build_lora_dataset.py")
    assert "random.Random(config.SAMPLE_SEED)" in text
    assert not re.search(r"\brandom\.seed\s*\(", text)
