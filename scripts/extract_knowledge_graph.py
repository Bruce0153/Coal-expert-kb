"""从标准 Document JSONL 快照抽取版本化知识图谱。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from coal_kb.research.graph_extraction import KnowledgeGraphExtractor


@dataclass
class ExtractKnowledgeGraph:
    """读取文档快照并写出单一 Graph Schema JSON。"""

    input_path: Path
    output_path: Path

    def process(self) -> dict[str, Any]:
        documents: list[Document] = []
        with self.input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Line {line_number} must be a JSON object")
                content = payload.get("page_content") or payload.get("content") or payload.get("text")
                if not isinstance(content, str):
                    raise ValueError(f"Line {line_number} is missing document text")
                metadata = payload.get("metadata") or {}
                if not isinstance(metadata, dict):
                    raise ValueError(f"Line {line_number} metadata must be an object")
                documents.append(Document(page_content=content, metadata=metadata))
        graph = KnowledgeGraphExtractor().process(documents)
        result = graph.to_dict()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.output_path)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a versioned knowledge graph from Document JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/artifacts/knowledge_graph.json")
    args = parser.parse_args()
    ExtractKnowledgeGraph(input_path=Path(args.input), output_path=Path(args.output)).process()


if __name__ == "__main__":
    main()

# 运行命令：PYTHONPATH=src python scripts/extract_knowledge_graph.py --input documents.jsonl
