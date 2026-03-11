from __future__ import annotations

import argparse
import logging
from typing import Optional

from coal_kb.cli_ui import print_banner, print_kv, print_stats_table
from coal_kb.logging import setup_logging
from coal_kb.qa.ask_pipeline import (
    HELP_TEXT,
    build_runtime,
    execute_query,
    format_claims,
    format_debug_info,
    format_source_cards,
    format_sources,
    log_query,
    parse_command,
)
from coal_kb.settings import load_config

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask the expert KB with a structured RAG pipeline.")
    parser.add_argument("query", nargs="*", help="Optional one-shot query. If omitted, starts interactive mode.")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--llm", action="store_true", help="Enable LLM answer generation.")
    parser.add_argument("--show-plan", action="store_true", help="Print the structured QueryPlan.")
    parser.add_argument("--save-trace", action="store_true", help="Persist plan and retrieval trace to the registry.")
    parser.add_argument("--debug", action="store_true", help="Show retrieval and context diagnostics.")
    parser.add_argument("--rerank-model", default=None, help="Override the reranker model.")
    parser.add_argument("--rerank-top-k", type=int, default=None)
    parser.add_argument("--llm-provider", default="none", choices=["none", "openai", "openai_compatible", "dashscope"])
    parser.add_argument("--backend", default=None, choices=["chroma", "elastic", "both"])
    parser.add_argument("--mode", default=None, choices=["strict", "balanced", "broad"])
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking for this session.")
    return parser


def _print_session_config(runtime, *, enable_llm: bool, debug: bool) -> None:
    print_banner("Coal KB Ask", f"backend={runtime.backend}")
    print_kv(
        "Retrieval Config",
        {
            "backend": runtime.backend,
            "k": str(runtime.k),
            "mode": runtime.mode,
            "rerank_enabled": str(runtime.retriever.rerank_enabled),
            "llm_answer": str(enable_llm),
            "debug": str(debug),
        },
    )


def _run_query(runtime, question: str, *, enable_llm: bool, show_plan: bool, debug: bool, save_trace: bool) -> None:
    execution = execute_query(runtime, question, enable_llm=enable_llm)

    if show_plan:
        print("\nQueryPlan:")
        print(execution.plan.to_json())

    print("\nAnswer:")
    print(execution.result.answer_text)

    if execution.result.claim_items:
        print("\nClaim Map:")
        print(format_claims(execution))

    if execution.result.citations:
        print("\nEvidence Catalog:")
        print(format_sources(execution))

    if execution.result.source_cards:
        print("\nSource Cards:")
        print(format_source_cards(execution))

    print_stats_table(
        "Query Stats",
        [
            ("docs", str(len(execution.docs))),
            ("evidence_sufficiency", execution.result.evidence_sufficiency),
            ("confidence", f"{execution.result.confidence_score:.2f}"),
            ("plan_ms", f"{execution.timings_ms['plan']:.2f}"),
            ("retrieve_ms", f"{execution.timings_ms['retrieve']:.2f}"),
            ("context_ms", f"{execution.timings_ms['context']:.2f}"),
            ("answer_ms", f"{execution.timings_ms['answer']:.2f}"),
            ("total_ms", f"{execution.timings_ms['total']:.2f}"),
        ],
    )

    if debug:
        print("\nDebug:")
        print(format_debug_info(execution))

    log_query(runtime, execution, save_trace=save_trace)


def _interactive_loop(runtime, *, enable_llm: bool, show_plan: bool, save_trace: bool, debug: bool) -> None:
    debug_enabled = debug
    print(HELP_TEXT)
    while True:
        try:
            raw_query = input("\nQuestion> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        command = parse_command(raw_query)
        if command == "exit":
            print("Bye.")
            return
        if command == "help":
            print(HELP_TEXT)
            continue
        if command == "debug":
            debug_enabled = not debug_enabled
            print(f"Debug mode: {debug_enabled}")
            continue

        if not raw_query.strip():
            continue

        try:
            _run_query(
                runtime,
                raw_query,
                enable_llm=enable_llm,
                show_plan=show_plan,
                debug=debug_enabled,
                save_trace=save_trace,
            )
        except Exception as exc:
            logger.exception("Ask pipeline failed")
            print(f"\nError: {exc}")


def main() -> None:
    args = _build_parser().parse_args()

    cfg = load_config()
    setup_logging(cfg, logger_name=__name__)

    runtime = build_runtime(
        cfg,
        backend=args.backend,
        k=args.k,
        rerank_enabled=False if args.no_rerank else None,
        rerank_top_n=args.rerank_top_k,
        rerank_model=args.rerank_model,
        mode=args.mode,
        enable_llm=args.llm,
        llm_provider=args.llm_provider,
    )
    _print_session_config(runtime, enable_llm=args.llm, debug=args.debug)

    one_shot_query: Optional[str] = " ".join(args.query).strip() or None
    if one_shot_query:
        _run_query(
            runtime,
            one_shot_query,
            enable_llm=args.llm,
            show_plan=args.show_plan,
            debug=args.debug,
            save_trace=args.save_trace,
        )
        return

    _interactive_loop(
        runtime,
        enable_llm=args.llm,
        show_plan=args.show_plan,
        save_trace=args.save_trace,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
