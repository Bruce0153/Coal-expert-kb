"""验证视觉索引只在显式配置后进入研究路线运行时。"""

from pathlib import Path

import pytest

from coal_kb.research.service import ResearchRouteService
from coal_kb.research.visual_assets import AssetType, MultimodalAsset, VisualAssetIndex


def _write_index(path: Path) -> None:
    asset = MultimodalAsset(
        asset_id="figure-1",
        asset_type=AssetType.IMAGE,
        source_file="paper.pdf",
        asset_path="figure.png",
        caption="Hydrogen yield figure",
    )
    VisualAssetIndex.build([asset]).write(path)


def test_visual_index_is_disabled_without_explicit_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COAL_KB_VISUAL_INDEX_PATH", raising=False)

    service = ResearchRouteService(standard_service=object())

    assert service.multimodal_route.visual_index is None


def test_visual_index_is_loaded_from_explicit_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_path = tmp_path / "visual_index.json"
    _write_index(index_path)
    monkeypatch.setenv("COAL_KB_VISUAL_INDEX_PATH", str(index_path))
    monkeypatch.setenv("COAL_KB_VISUAL_TOP_K", "2")
    monkeypatch.setenv("COAL_KB_VISUAL_WEIGHT", "0.8")

    service = ResearchRouteService(standard_service=object())

    assert service.multimodal_route.visual_index is not None
    assert service.multimodal_route.visual_top_k == 2
    assert service.multimodal_route.visual_weight == 0.8


def test_missing_explicit_visual_index_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COAL_KB_VISUAL_INDEX_PATH", str(tmp_path / "missing.json"))

    with pytest.raises(FileNotFoundError, match="visual asset index"):
        ResearchRouteService(standard_service=object())
