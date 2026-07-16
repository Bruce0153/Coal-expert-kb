"""验证资产 Manifest、视觉索引和 Multimodal route 资产扩展。"""

from __future__ import annotations

import base64
from pathlib import Path

import fitz
from langchain_core.documents import Document

from coal_kb.research import MultimodalRoute, RouteResult
from coal_kb.research.visual_assets import (
    AssetManifest,
    AssetType,
    MultimodalAsset,
    MultimodalAssetExtractor,
    VisualAssetIndex,
)

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z1ZQAAAAASUVORK5CYII="
)


def test_pdf_asset_extraction_writes_stable_manifest(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_image(fitz.Rect(50, 50, 150, 150), stream=_PNG)
    page.insert_text((50, 175), "Figure 1 Hydrogen yield curve")
    document.save(pdf_path)
    document.close()

    manifest = MultimodalAssetExtractor(
        output_dir=tmp_path / "assets",
        extract_tables=False,
    ).process([pdf_path])
    loaded = AssetManifest.load(tmp_path / "assets" / "manifest.jsonl")

    assert len(manifest.assets) == 1
    assert loaded.assets[0].asset_type is AssetType.IMAGE
    assert loaded.assets[0].page == 1
    assert Path(loaded.assets[0].asset_path).is_file()
    assert "Hydrogen yield" in loaded.assets[0].caption


def test_visual_index_round_trip_and_multimodal_route_expansion(tmp_path: Path) -> None:
    asset = MultimodalAsset(
        asset_id="figure-hydrogen",
        asset_type=AssetType.IMAGE,
        source_file="paper.pdf",
        asset_path="assets/figure.png",
        page=3,
        caption="Figure 2 Hydrogen yield versus steam ratio",
    )
    index = VisualAssetIndex.build([asset])
    index_path = tmp_path / "visual_index.json"
    index.write(index_path)
    loaded = VisualAssetIndex.load(index_path)

    result = MultimodalRoute(visual_index=loaded, visual_top_k=1).process(
        "Show the hydrogen yield figure",
        lambda: RouteResult(
            documents=[Document(page_content="Text evidence", metadata={"chunk_id": "text-1"})],
            trace={},
        ),
    )

    assert result.documents[0].metadata["asset_id"] == "figure-hydrogen"
    assert result.documents[0].metadata["research_modality"] == "image"
    visual = result.trace["multimodal"]["visual_retrieval"]
    assert visual["enabled"] is True
    assert visual["result_count"] == 1
    assert visual["results"][0]["page"] == 3


def test_external_visual_index_requires_compatible_query_encoder() -> None:
    asset = MultimodalAsset(
        asset_id="asset-1",
        asset_type=AssetType.IMAGE,
        source_file="paper.pdf",
        asset_path="asset.png",
        caption="gasification diagram",
    )
    index = VisualAssetIndex.build([asset], embedding_fn=lambda _: [[1.0, 0.0]])

    try:
        index.search("diagram")
    except ValueError as exc:
        assert "compatible query embedding" in str(exc)
    else:
        raise AssertionError("External visual index must require a query embedding function")
