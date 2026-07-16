"""抽取、记录并检索图像与表格资产。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from langchain_core.documents import Document

ASSET_MANIFEST_VERSION = "multimodal-assets.v1"
VISUAL_INDEX_VERSION = "visual-asset-index.v1"
_SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_CAPTION_RE = re.compile(r"^(?:fig(?:ure)?\.?|图|表|table|chart)\s*\d*", re.IGNORECASE)


class AssetType(str, Enum):
    """多模态资产类型。"""

    IMAGE = "image"
    TABLE = "table"


@dataclass(frozen=True)
class MultimodalAsset:
    """保存资产文件、页面定位、描述文本和来源。"""

    asset_id: str
    asset_type: AssetType
    source_file: str
    asset_path: str
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    caption: str = ""
    extracted_text: str = ""
    mime_type: str = "application/octet-stream"
    checksum: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def search_text(self) -> str:
        values = [self.caption, self.extracted_text, Path(self.source_file).stem]
        return "\n".join(value.strip() for value in values if value and value.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "asset_type": self.asset_type.value,
            "source_file": self.source_file,
            "asset_path": self.asset_path,
            "page": self.page,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "caption": self.caption,
            "extracted_text": self.extracted_text,
            "mime_type": self.mime_type,
            "checksum": self.checksum,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MultimodalAsset:
        bbox = payload.get("bbox")
        return cls(
            asset_id=str(payload["asset_id"]),
            asset_type=AssetType(str(payload["asset_type"])),
            source_file=str(payload["source_file"]),
            asset_path=str(payload["asset_path"]),
            page=int(payload["page"]) if payload.get("page") is not None else None,
            bbox=tuple(float(value) for value in bbox) if bbox is not None else None,
            caption=str(payload.get("caption") or ""),
            extracted_text=str(payload.get("extracted_text") or ""),
            mime_type=str(payload.get("mime_type") or "application/octet-stream"),
            checksum=str(payload.get("checksum") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_document(self, score: float | None = None) -> Document:
        metadata = {
            **self.metadata,
            "modality": self.asset_type.value,
            "research_modality": self.asset_type.value,
            "asset_id": self.asset_id,
            "asset_path": self.asset_path,
            "image_path": self.asset_path if self.asset_type is AssetType.IMAGE else None,
            "table_id": self.asset_id if self.asset_type is AssetType.TABLE else None,
            "source_file": self.source_file,
            "page": self.page,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "caption": self.caption,
        }
        if score is not None:
            metadata["visual_retrieval_score"] = round(score, 6)
        return Document(page_content=self.search_text or self.asset_id, metadata=metadata)


@dataclass
class AssetManifest:
    """持久化版本化资产清单。"""

    assets: list[MultimodalAsset]
    version: str = ASSET_MANIFEST_VERSION

    def write(self, output_dir: Path) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "manifest.jsonl"
        jsonl_path.write_text(
            "\n".join(json.dumps(asset.to_dict(), ensure_ascii=False, sort_keys=True) for asset in self.assets)
            + ("\n" if self.assets else ""),
            encoding="utf-8",
        )
        summary = {
            "manifest_version": self.version,
            "asset_count": len(self.assets),
            "types": {
                asset_type.value: sum(asset.asset_type is asset_type for asset in self.assets)
                for asset_type in AssetType
            },
            "manifest_path": str(jsonl_path),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary

    @classmethod
    def load(cls, path: Path) -> AssetManifest:
        assets: list[MultimodalAsset] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise ValueError("Each asset manifest line must be a JSON object")
                    assets.append(MultimodalAsset.from_dict(payload))
        return cls(assets=assets)


def _checksum(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _asset_id(source: str, page: int | None, kind: str, checksum: str, occurrence: int) -> str:
    value = f"{source}|{page}|{kind}|{checksum}|{occurrence}"
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _mime_for_extension(extension: str) -> str:
    extension = extension.lower().lstrip(".")
    aliases = {"jpg": "jpeg", "tif": "tiff"}
    normalized = aliases.get(extension, extension)
    return f"image/{normalized}" if normalized else "application/octet-stream"


@dataclass
class MultimodalAssetExtractor:
    """从 PDF、独立图片和 PyMuPDF 表格中抽取资产。"""

    output_dir: Path
    extract_tables: bool = True

    def process(self, inputs: Iterable[Path]) -> AssetManifest:
        assets: list[MultimodalAsset] = []
        for path in self._expand_inputs(inputs):
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                assets.extend(self._extract_pdf(path))
            elif suffix in _SUPPORTED_IMAGES:
                assets.append(self._extract_image_file(path))
        manifest = AssetManifest(assets=sorted(assets, key=lambda item: item.asset_id))
        manifest.write(self.output_dir)
        return manifest

    @staticmethod
    def _expand_inputs(inputs: Iterable[Path]) -> list[Path]:
        paths: list[Path] = []
        for raw in inputs:
            path = raw.resolve()
            if path.is_dir():
                paths.extend(
                    candidate
                    for candidate in sorted(path.rglob("*"))
                    if candidate.is_file()
                    and (candidate.suffix.lower() == ".pdf" or candidate.suffix.lower() in _SUPPORTED_IMAGES)
                )
            elif path.is_file():
                paths.append(path)
        return sorted(set(paths))

    def _extract_image_file(self, path: Path) -> MultimodalAsset:
        content = path.read_bytes()
        checksum = _checksum(content)
        asset_id = _asset_id(str(path), None, "image", checksum, 0)
        destination = self.output_dir / "assets" / f"{asset_id}{path.suffix.lower()}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return MultimodalAsset(
            asset_id=asset_id,
            asset_type=AssetType.IMAGE,
            source_file=str(path),
            asset_path=str(destination),
            caption=path.stem.replace("_", " "),
            mime_type=_mime_for_extension(path.suffix),
            checksum=checksum,
            metadata={"extractor": "standalone_image_v1"},
        )

    def _extract_pdf(self, path: Path) -> list[MultimodalAsset]:
        import fitz

        assets: list[MultimodalAsset] = []
        with fitz.open(path) as document:
            for page_index, page in enumerate(document):
                blocks = page.get_text("blocks")
                for occurrence, image in enumerate(page.get_images(full=True)):
                    xref = int(image[0])
                    extracted = document.extract_image(xref)
                    content = bytes(extracted["image"])
                    extension = str(extracted.get("ext") or "png")
                    checksum = _checksum(content)
                    rects = page.get_image_rects(xref)
                    rect = rects[0] if rects else None
                    asset_id = _asset_id(str(path), page_index + 1, "image", checksum, occurrence)
                    destination = self.output_dir / "assets" / f"{asset_id}.{extension}"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(content)
                    bbox = tuple(float(value) for value in rect) if rect is not None else None
                    assets.append(
                        MultimodalAsset(
                            asset_id=asset_id,
                            asset_type=AssetType.IMAGE,
                            source_file=str(path),
                            asset_path=str(destination),
                            page=page_index + 1,
                            bbox=bbox,
                            caption=self._caption(blocks, bbox),
                            mime_type=_mime_for_extension(extension),
                            checksum=checksum,
                            metadata={"extractor": "pymupdf_image_v1", "xref": xref},
                        )
                    )
                if self.extract_tables:
                    assets.extend(self._extract_tables(path, page, page_index + 1))
        return assets

    def _extract_tables(self, source_path: Path, page: Any, page_number: int) -> list[MultimodalAsset]:
        if not hasattr(page, "find_tables"):
            return []
        try:
            finder = page.find_tables()
        except Exception:
            return []
        assets: list[MultimodalAsset] = []
        for occurrence, table in enumerate(getattr(finder, "tables", [])):
            rows = table.extract()
            text = "\n".join(" | ".join(str(cell or "") for cell in row) for row in rows)
            content = json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")
            checksum = _checksum(content)
            asset_id = _asset_id(str(source_path), page_number, "table", checksum, occurrence)
            destination = self.output_dir / "assets" / f"{asset_id}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            bbox = tuple(float(value) for value in table.bbox)
            assets.append(
                MultimodalAsset(
                    asset_id=asset_id,
                    asset_type=AssetType.TABLE,
                    source_file=str(source_path),
                    asset_path=str(destination),
                    page=page_number,
                    bbox=bbox,
                    caption=self._caption(page.get_text("blocks"), bbox),
                    extracted_text=text,
                    mime_type="application/json",
                    checksum=checksum,
                    metadata={"extractor": "pymupdf_table_v1", "row_count": len(rows)},
                )
            )
        return assets

    @staticmethod
    def _caption(blocks: Sequence[Any], bbox: tuple[float, float, float, float] | None) -> str:
        candidates: list[tuple[int, float, str]] = []
        for block in blocks:
            if len(block) < 5:
                continue
            text = " ".join(str(block[4]).split()).strip()
            if not text:
                continue
            if bbox is None:
                distance = float(block[1])
            else:
                x0, y0, x1, _ = (float(block[index]) for index in range(4))
                horizontal_overlap = max(0.0, min(x1, bbox[2]) - max(x0, bbox[0]))
                if horizontal_overlap <= 0 or y0 < bbox[1] - 20 or y0 > bbox[3] + 180:
                    continue
                distance = abs(y0 - bbox[3])
            priority = 0 if _CAPTION_RE.search(text) else 1
            candidates.append((priority, distance, text))
        return min(candidates, default=(1, 0.0, ""))[2]


def _tokens(text: str) -> list[str]:
    lowered = text.casefold()
    tokens = re.findall(r"[a-z0-9][a-z0-9_-]{1,}", lowered)
    chinese = re.findall(r"[\u4e00-\u9fff]+", lowered)
    for segment in chinese:
        tokens.extend(segment[index : index + 2] for index in range(max(1, len(segment) - 1)))
    return tokens


def _lexical_vector(text: str, dimension: int) -> list[float]:
    vector = [0.0] * dimension
    for token in _tokens(text):
        index = int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16) % dimension
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


@dataclass(frozen=True)
class VisualSearchResult:
    """保存资产检索分数和资产内容。"""

    asset: MultimodalAsset
    score: float


AssetEmbeddingFunction = Callable[[list[MultimodalAsset]], list[list[float]]]
QueryEmbeddingFunction = Callable[[list[str]], list[list[float]]]


@dataclass
class VisualAssetIndex:
    """持久化 caption-first 或外部视觉编码器生成的资产向量。"""

    assets: list[MultimodalAsset]
    vectors: list[list[float]]
    dimension: int
    encoder: str = "lexical-caption-v1"
    version: str = VISUAL_INDEX_VERSION

    @classmethod
    def build(
        cls,
        assets: Iterable[MultimodalAsset],
        *,
        embedding_fn: AssetEmbeddingFunction | None = None,
        dimension: int = 256,
    ) -> VisualAssetIndex:
        items = list(assets)
        vectors = embedding_fn(items) if embedding_fn is not None else [_lexical_vector(item.search_text, dimension) for item in items]
        if len(vectors) != len(items):
            raise ValueError("Visual embedding function must return one vector per asset")
        active_dimension = len(vectors[0]) if vectors else dimension
        if any(len(vector) != active_dimension for vector in vectors):
            raise ValueError("Visual index vectors must share one dimension")
        return cls(
            assets=items,
            vectors=[[float(value) for value in vector] for vector in vectors],
            dimension=active_dimension,
            encoder="external" if embedding_fn is not None else "lexical-caption-v1",
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "index_version": self.version,
            "dimension": self.dimension,
            "encoder": self.encoder,
            "entries": [
                {"asset": asset.to_dict(), "vector": vector}
                for asset, vector in zip(self.assets, self.vectors)
            ],
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> VisualAssetIndex:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("index_version") != VISUAL_INDEX_VERSION:
            raise ValueError(f"Unsupported visual index version: {payload.get('index_version')}")
        entries = payload.get("entries") or []
        return cls(
            assets=[MultimodalAsset.from_dict(entry["asset"]) for entry in entries],
            vectors=[[float(value) for value in entry["vector"]] for entry in entries],
            dimension=int(payload["dimension"]),
            encoder=str(payload.get("encoder") or "unknown"),
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        modalities: set[AssetType] | None = None,
        query_embedding_fn: QueryEmbeddingFunction | None = None,
    ) -> list[VisualSearchResult]:
        if top_k < 1:
            return []
        if self.encoder == "external" and query_embedding_fn is None:
            raise ValueError("External visual index requires a compatible query embedding function")
        query_vector = (
            query_embedding_fn([query])[0]
            if query_embedding_fn is not None
            else _lexical_vector(query, self.dimension)
        )
        if len(query_vector) != self.dimension:
            raise ValueError("Query embedding dimension does not match the visual index")
        results = [
            VisualSearchResult(asset=asset, score=round(_cosine(query_vector, vector), 8))
            for asset, vector in zip(self.assets, self.vectors)
            if modalities is None or asset.asset_type in modalities
        ]
        return sorted(results, key=lambda result: (-result.score, result.asset.asset_id))[:top_k]
