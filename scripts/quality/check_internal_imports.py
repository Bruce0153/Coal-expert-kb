"""验证仓库内部 import 均指向现有正式模块。"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "coal_kb"
SCAN_ROOTS = (PACKAGE_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "tests")


def _module_exists(module_name: str) -> bool:
    if not module_name.startswith("coal_kb"):
        return True
    relative = Path(*module_name.split("."))
    source_root = REPO_ROOT / "src"
    return (source_root / relative).with_suffix(".py").is_file() or (
        source_root / relative / "__init__.py"
    ).is_file()


def _package_for(path: Path) -> list[str]:
    relative = path.relative_to(REPO_ROOT / "src").with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    else:
        parts.pop()
    return parts


def _resolve_relative(path: Path, node: ast.ImportFrom) -> str | None:
    if not path.is_relative_to(PACKAGE_ROOT):
        return None
    package = _package_for(path)
    parent_hops = max(0, node.level - 1)
    if parent_hops > len(package):
        return None
    base = package[: len(package) - parent_hops]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


@dataclass
class CheckInternalImports:
    """扫描 Python 文件并报告失效的仓库内部 import。"""

    def process(self) -> list[str]:
        violations: list[str] = []
        for root in SCAN_ROOTS:
            for path in sorted(root.rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    modules: list[str] = []
                    if isinstance(node, ast.Import):
                        modules.extend(alias.name for alias in node.names if alias.name.startswith("coal_kb"))
                    elif isinstance(node, ast.ImportFrom):
                        if node.level:
                            resolved = _resolve_relative(path, node)
                            if resolved:
                                modules.append(resolved)
                        elif node.module and node.module.startswith("coal_kb"):
                            modules.append(node.module)
                            if node.module == "coal_kb":
                                modules.extend(
                                    f"coal_kb.{alias.name}"
                                    for alias in node.names
                                    if alias.name.islower()
                                )
                    for module_name in modules:
                        if not _module_exists(module_name):
                            relative = path.relative_to(REPO_ROOT)
                            violations.append(f"{relative}:{node.lineno}: {module_name}")
        if violations:
            raise RuntimeError("Invalid internal imports:\n" + "\n".join(violations))
        return violations


def main() -> None:
    CheckInternalImports().process()


if __name__ == "__main__":
    main()

# 运行命令：PYTHONPATH=src python scripts/quality/check_internal_imports.py
