"""抽取多模态资产并构建可持久化视觉资产索引。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coal_kb.research.visual_assets import MultimodalAssetExtractor, VisualAssetIndex


@dataclass
class ExtractMultimodalAssets:
    """执行资产抽取、Manifest 写出和 caption-first 索引构建。"""

    inputs: list[Path]
    output_dir: Path
    index_path: Path
    extract_tables: bool = True

    def process(self) -> dict[str, Any]:
        manifest = MultimodalAssetExtractor(
            output_dir=self.output_dir,
            extract_tables=self.extract_tables,
        ).process(self.inputs)
        index = VisualAssetIndex.build(manifest.assets)
        index.write(self.index_path)
        return {
            "asset_count": len(manifest.assets),
            "manifest_path": str(self.output_dir / "manifest.jsonl"),
            "index_path": str(self.index_path),
            "encoder": index.encoder,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract multimodal assets and build a visual asset index.")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-dir", default="data/artifacts/multimodal_assets")
    parser.add_argument("--index", default="data/artifacts/multimodal_assets/visual_index.json")
    parser.add_argument("--no-tables", action="store_true")
    args = parser.parse_args()
    ExtractMultimodalAssets(
        inputs=[Path(value) for value in args.inputs],
        output_dir=Path(args.output_dir),
        index_path=Path(args.index),
        extract_tables=not args.no_tables,
    ).process()


if __name__ == "__main__":
    main()

# 运行命令：PYTHONPATH=src python scripts/extract_multimodal_assets.py data/raw_pdfs
