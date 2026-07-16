"""启动命令行交互式检索与证据回答会话。"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

from coal_kb.application.ask import (
    AskRuntime,
    build_response_payload,
    build_runtime,
    execute_query,
    log_query,
    parse_command,
)
from coal_kb.infra.config import AppConfig, load_config
from coal_kb.infra.observability.logging import setup_logging
from coal_kb.interfaces.cli import print_banner, print_kv, print_stats_table
from coal_kb.research import ResearchRoute

logger = logging.getLogger(__name__)


@dataclass
class Ask:
    cfg: AppConfig
    args: argparse.Namespace

    def _build_runtime(self) -> AskRuntime:
        runtime = build_runtime(
            self.cfg,
            backend=self.args.backend,
            k=self.args.k,
            rerank_enabled=False if self.args.no_rerank else None,
            rerank_top_n=self.args.rerank_top_k,
            mode=self.args.mode,
            enable_llm=self.args.llm,
            llm_provider=self.args.llm_provider,
        )
        print_kv(
            "Retrieval Config",
            {
                "backend": runtime.backend,
                "k": runtime.k,
                "rerank_enabled": runtime.retriever.rerank_enabled,
                "rerank_top_n": runtime.retriever.rerank_top_n,
                "max_per_source": self.cfg.retrieval.max_per_source,
                "mode": runtime.mode,
                "research_route": self.args.research_route,
            },
        )
        return runtime

    @staticmethod
    def _print_response(payload: dict[str, object]) -> None:
        print(f"\n{payload['answer']}")
        citations = payload.get("citations") or []
        if citations:
            print("\n引用列表:")
            for item in citations:
                if not isinstance(item, dict):
                    continue
                print(
                    f"- [{item.get('label')}] {item.get('source_file', 'unknown')}"
                    f" | page={item.get('page')}"
                    f" | heading={item.get('heading_path')}"
                    f" | chunk={item.get('chunk_id')}"
                )

    def process(self) -> None:
        runtime = self._build_runtime()
        debug = self.args.debug
        print_banner("Coal KB Ask", f"backend={runtime.backend}")
        print("输入 help 查看命令，输入 exit 退出。")
        while True:
            question = input("\n你的问题> ").strip()
            if not question:
                continue
            command = parse_command(question)
            if command == "exit":
                return
            if command == "help":
                print("help：显示帮助；debug：切换调试输出；exit：退出。")
                continue
            if command == "debug":
                debug = not debug
                print(f"debug={debug}")
                continue
            try:
                execution = execute_query(
                    runtime,
                    question,
                    enable_llm=self.args.llm,
                    research_route=self.args.research_route,
                )
                if self.args.show_plan:
                    print("\nQueryPlan:")
                    print(execution.plan.to_json())
                log_query(runtime, execution, save_trace=self.args.save_trace or debug)
                payload = build_response_payload(execution, include_debug=debug)
                print_stats_table(
                    "Query Stats",
                    [
                        ("docs", len(execution.docs)),
                        ("latency_ms", execution.timings_ms["total"]),
                        ("evidence", execution.result.evidence_sufficiency),
                        ("confidence", execution.result.confidence_score),
                        ("research_route", execution.research_route),
                    ],
                )
                self._print_response(payload)
                if debug:
                    print("\nDiagnostics:")
                    print(payload["diagnostics"])
            except Exception as error:
                print(f"\n检索或回答失败: {type(error).__name__}: {error}")
                logger.exception("Ask loop failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the expert KB with metadata-aware retrieval.")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--llm", action="store_true", help="Enable LLM answer generation.")
    parser.add_argument("--show-plan", action="store_true", help="Print QueryPlan JSON.")
    parser.add_argument("--save-trace", action="store_true", help="Persist retrieval trace.")
    parser.add_argument("--debug", action="store_true", help="Print full diagnostics.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable configured reranking.")
    parser.add_argument("--rerank-top-k", type=int, default=None)
    parser.add_argument(
        "--research-route",
        choices=[route.value for route in ResearchRoute],
        default=ResearchRoute.STANDARD.value,
    )
    parser.add_argument(
        "--llm-provider",
        default="none",
        choices=["none", "openai", "openai_compatible", "dashscope"],
    )
    parser.add_argument("--backend", default=None, choices=["chroma", "elastic", "both"])
    parser.add_argument("--mode", default=None, choices=["strict", "balanced", "broad"])
    args = parser.parse_args()
    cfg = load_config()
    setup_logging(cfg, logger_name=__name__)
    Ask(cfg=cfg, args=args).process()


if __name__ == "__main__":
    main()

# 运行命令：PYTHONPATH=src python scripts/ask.py --research-route graph
