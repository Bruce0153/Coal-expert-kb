"""验证项目依赖声明与 Harness 安装清单保持一致。"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
CI_REQUIREMENTS_PATH = REPO_ROOT / "requirements" / "ci.txt"


def _requirement_name(requirement: str) -> str:
    name = re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip()
    return name.lower().replace("_", "-")


def _requirements(lines: list[str]) -> set[str]:
    return {
        _requirement_name(line)
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    }


@dataclass
class CheckDependencies:
    """检查 CI 依赖是否已在 pyproject 中声明且没有重复。"""

    def process(self) -> dict[str, int]:
        project = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))["project"]
        declared_lines = list(project.get("dependencies", []))
        for values in project.get("optional-dependencies", {}).values():
            declared_lines.extend(values)
        declared = _requirements(declared_lines)

        ci_lines = CI_REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
        ci_names = [_requirement_name(line) for line in ci_lines if line.strip() and not line.lstrip().startswith("#")]
        duplicates = sorted({name for name in ci_names if ci_names.count(name) > 1})
        undeclared = sorted(set(ci_names) - declared)
        if duplicates or undeclared:
            details = []
            if duplicates:
                details.append(f"duplicate CI dependencies: {duplicates}")
            if undeclared:
                details.append(f"CI dependencies missing from pyproject.toml: {undeclared}")
            raise RuntimeError("; ".join(details))

        dev = _requirements(list(project.get("optional-dependencies", {}).get("dev", [])))
        missing_dev = sorted(dev - set(ci_names))
        if missing_dev:
            raise RuntimeError(f"Harness is missing dev dependencies: {missing_dev}")
        return {"declared": len(declared), "ci": len(ci_names), "dev": len(dev)}


def main() -> None:
    CheckDependencies().process()


if __name__ == "__main__":
    main()

# 运行命令：python scripts/quality/check_dependencies.py
