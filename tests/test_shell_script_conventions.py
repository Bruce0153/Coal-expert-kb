"""验证 Shell 运行入口仅负责编排并正确加载 config.sh。"""

from __future__ import annotations

from pathlib import Path


def test_shell_runners_source_local_config_and_use_set_e() -> None:
    runners = sorted(Path("scripts").glob("*/run_*.sh"))
    assert runners
    for path in runners:
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "#!/usr/bin/env bash", path
        assert "set -e" in lines[:8], path
        assert 'SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)' in lines, path
        assert 'source "$SCRIPT_DIR/config.sh"' in lines, path
        assert lines[-1].startswith("# 运行命令："), path


def test_shell_configs_only_define_referenced_variables() -> None:
    for config_path in sorted(Path("scripts").glob("*/config.sh")):
        runner_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in config_path.parent.glob("run_*.sh")
        )
        assignments = []
        for line in config_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("set "):
                continue
            if "=" in stripped and not stripped.startswith("SCRIPT_DIR="):
                assignments.append(stripped.split("=", 1)[0])
        for variable in assignments:
            if variable == "PROJECT_ROOT":
                assert "$PROJECT_ROOT" in config_path.read_text(encoding="utf-8")
            else:
                assert f"${variable}" in runner_text or f"${{{variable}}}" in runner_text, (
                    config_path,
                    variable,
                )
