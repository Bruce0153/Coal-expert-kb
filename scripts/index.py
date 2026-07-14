"""构建、切换或回滚 Elasticsearch 物理索引。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from coal_kb.indexing.service import IndexService
from coal_kb.infra.config import AppConfig, load_config
from coal_kb.infra.observability.logging import setup_logging
from coal_kb.interfaces.cli.ui import print_banner, print_kv, print_stats_table


@dataclass
class Index:
    """保存命令行参数并调用索引应用服务。"""

    cfg: AppConfig
    args: argparse.Namespace

    def _print_build_result(self, result: dict[str, Any]) -> None:
        stats = result["stats"]
        validation = result["validation"]
        print_stats_table(
            "Build Summary",
            [
                ("index", str(result["index"])),
                ("indexed", str(stats.get("indexed"))),
                ("chunks", str(stats.get("chunks"))),
                ("doc_types", str(stats.get("doc_type_counts"))),
                ("languages", str(stats.get("language_counts"))),
                ("validated", str(validation.get("ok"))),
                ("elapsed_s", str(stats.get("elapsed_s"))),
            ],
        )

    def process(self) -> dict[str, Any]:
        print_banner("Coal KB Index Manager", f"backend={self.cfg.backend}")
        service = IndexService(self.cfg)
        if self.args.cmd == "build":
            print_kv(
                "Index Build",
                {
                    "embedding_version": self.args.embedding_version or self.cfg.model_versions.embedding_version,
                    "index_prefix": self.cfg.elastic.index_prefix,
                    "alias_current": self.cfg.elastic.alias_current,
                    "alias_prev": self.cfg.elastic.alias_prev,
                },
            )
            result = service.process(
                "build",
                embedding_version=self.args.embedding_version,
                resume_index=self.args.resume_index,
            )
            self._print_build_result(result)
            return result
        if self.args.cmd == "switch":
            result = service.process("switch", index_name=self.args.index)
            print_stats_table(
                "Alias Switch",
                [("alias_current", str(result["alias_current"])), ("new_index", str(result["new_index"]))],
            )
            return result
        if self.args.cmd == "rollback":
            result = service.process("rollback")
            print_stats_table(
                "Alias Rollback",
                [("alias_current", str(result["alias_current"])), ("alias_prev", str(result["alias_prev"]))],
            )
            return result
        raise ValueError(f"Unsupported command: {self.args.cmd}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Elasticsearch index versions.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    build = subparsers.add_parser("build", help="Create new index and ingest with elastic backend.")
    build.add_argument("--embedding-version", default=None, help="Override embedding version.")
    build.add_argument("--resume-index", default=None, help="Continue writing into an existing physical index.")
    switch = subparsers.add_parser("switch", help="Switch alias_current to a specific index.")
    switch.add_argument("--index", required=True, help="Target index name.")
    subparsers.add_parser("rollback", help="Rollback alias_current to alias_prev.")
    args = parser.parse_args()
    cfg = load_config()
    setup_logging(cfg, logger_name=__name__)
    Index(cfg=cfg, args=args).process()


if __name__ == "__main__":
    main()

# 运行命令：python scripts/index.py build
