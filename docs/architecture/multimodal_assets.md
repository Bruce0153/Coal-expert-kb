# 多模态资产抽取与视觉检索

多模态能力分为资产层、索引层和路线层。默认问答行为不加载视觉索引；只有显式配置后，Multimodal 和 Agent 路线才会把资产级证据加入候选集合。

## 资产抽取

`MultimodalAssetExtractor` 支持：

- PDF 内嵌图像：保存原始图像字节、页码、边界框、校验和和附近图注；
- PDF 表格：在当前 PyMuPDF 支持 `find_tables()` 时保存结构化 JSON、行文本、页码和边界框；
- 独立图片：支持 PNG、JPEG、WebP、BMP 和 TIFF。

每个资产使用来源、页码、类型、内容校验和和出现序号生成稳定 ID。输出目录包含 `assets/`、`manifest.jsonl` 和版本化 `manifest.json`。

```bash
PYTHONPATH=src python scripts/extract_multimodal_assets.py \
  data/raw_pdfs \
  --output-dir data/artifacts/multimodal_assets \
  --index data/artifacts/multimodal_assets/visual_index.json
```

## 视觉资产索引

默认索引使用图注、表格文本和来源文件名构建确定性 caption-first 向量，因此离线可运行且不会下载模型。`VisualAssetIndex.build()` 也接受外部资产编码器；外部索引在查询时必须提供维度兼容的查询编码器，禁止混用向量空间。

索引保存资产完整来源、向量维度、编码器名称和版本。检索可限制为 `image` 或 `table`，返回资产、分数、来源文件和页码。

## 路线接入

显式设置以下环境变量后，`ResearchRouteService` 会加载视觉索引：

```bash
export COAL_KB_VISUAL_INDEX_PATH=data/artifacts/multimodal_assets/visual_index.json
export COAL_KB_VISUAL_TOP_K=5
export COAL_KB_VISUAL_WEIGHT=1.0
```

未配置 `COAL_KB_VISUAL_INDEX_PATH` 时，现有 Standard、Graph、Multimodal 和 Agent 行为保持不变。配置路径不存在、Top-K 非正数或权重为负数时直接报错，不静默回退。

Multimodal route 会：

1. 执行标准证据路线；
2. 根据问题中的图像或表格意图查询资产索引；
3. 将资产转为标准 `Document`，携带 `asset_id`、`asset_path`、页码、边界框、图注和检索分数；
4. 与基础证据去重并统一重排；
5. 在 Trace 中记录索引编码器、资产结果和最终模态分布。
