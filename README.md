# Coal Expert KB

Coal Expert KB 是面向煤热解、气化与燃烧文献的证据驱动知识库。系统将论文、报告和结构化数据转换为可检索证据，并通过统一的查询规划、检索、上下文构建和回答链路提供带引用的科研问答。

## 核心能力

- 摄入 PDF、Markdown、文本、HTML、DOCX、PPTX、CSV、XLSX、JSON 和 JSONL。
- 按标题层级和语义边界构建父子 Chunk，并保留来源、页码、章节和父子关系。
- 支持 Chroma、Elasticsearch 和二者的混合检索。
- 支持元数据约束、两阶段父子检索、RRF、Rerank 和来源多样性控制。
- 支持事实、条件、比较、多跳、统计聚合、表格和跨文档问题。
- 基于 SQLite 保存结构化实验记录、证据、冲突信息、查询日志和会话记录。
- 为 CLI 和 HTTP API 提供一致的回答、引用、证据卡片和诊断信息。
- 提供离线 Evaluation Pipeline 与统一仓库 Harness。

## 运行链路

```text
Document
  → Loader / Parser
  → Metadata normalization
  → Hierarchical chunking
  → Chroma / Elasticsearch / SQLite

Question
  → QueryPlanner
  → ConstraintSet + ComplexQuestionSpec
  → Recall / Rerank / Aggregation / Table retrieval
  → ContextBuilder
  → Answerer
  → Claims + Citations + Source cards
```

所有问答路线最终都返回标准 `Document` 证据，并复用同一套上下文、引用和回答实现。

## 环境要求

- Python 3.10 或更高版本
- Elasticsearch 8.x，仅在使用 Elasticsearch 后端时需要
- 远程模型 API Key，仅在对应 Provider 使用 `remote` 模式时需要
- 本地模型文件或可访问的模型仓库，仅在 Provider 使用 `local` 模式时需要

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

需要处理 Office 文档时：

```bash
python -m pip install -e ".[docs]"
```

需要训练结构化抽取模型时：

```bash
python -m pip install -e ".[training]"
```

开发和测试环境：

```bash
python -m pip install -e ".[dev,docs]"
```

## 配置

主配置文件为 `configs/app.yaml`。也可以通过 `COAL_KB_CONFIG` 指向其他配置文件。

Provider 使用明确的远程或本地模式，不会在远程调用失败后自动切换到本地实现。

```yaml
backend: elastic

embeddings:
  mode: remote
  remote:
    provider: dashscope
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: "DASHSCOPE_API_KEY"
    model: "text-embedding-v4"
    dimensions: 1024
  local:
    provider: huggingface
    model: "BAAI/bge-m3"
    model_path: null
    device: auto
    dimensions: 1024

rerank:
  mode: remote
  remote:
    provider: dashscope
    base_url: "https://dashscope.aliyuncs.com/api/v1/services/rerank"
    api_key_env: "DASHSCOPE_API_KEY"
    model: "qwen3-rerank"
    endpoint: "/text-rerank/text-rerank"
  local:
    provider: cross_encoder
    model: "BAAI/bge-reranker-base"
    model_path: null
    device: auto

llm:
  mode: remote
  temperature: 0.0
  remote:
    provider: dashscope
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: "DASHSCOPE_API_KEY"
    model: "qwen3.5-flash"
  local:
    provider: vllm
    base_url: "http://127.0.0.1:8001/v1"
    model: "Qwen/Qwen3-8B"
    model_path: null
    device: auto
```

远程密钥通过环境变量提供：

```bash
export DASHSCOPE_API_KEY="..."
```

常用环境覆盖：

```bash
export COAL_KB_CONFIG="configs/app.yaml"
export COAL_KB_CHROMA_DIR="storage/chroma_db"
export COAL_KB_SQLITE_PATH="storage/expert.db"
export COAL_KB_LOG_LEVEL="INFO"
```

## 数据目录

```text
data/
├── raw_pdfs/       # 原始 PDF
├── raw_docs/       # 其他原始文档
├── interim/        # 中间产物和标准表格记录
├── artifacts/      # 数据集、模型和评估产物
└── eval/           # 评估数据集

storage/
├── chroma_db/      # Chroma 持久化目录
├── expert.db       # 结构化实验记录
└── kb.db           # 查询和会话注册库
```

## 文档摄入与索引

将文件放入 `data/raw_pdfs/` 或 `data/raw_docs/` 后运行：

```bash
PYTHONPATH=src python scripts/ingest.py
PYTHONPATH=src python scripts/index.py
PYTHONPATH=src python scripts/validate_index.py
```

重建摄入产物或启用表格抽取：

```bash
PYTHONPATH=src python scripts/ingest.py --rebuild
PYTHONPATH=src python scripts/ingest.py --tables --table-flavor auto
```

也可以使用 Shell 入口：

```bash
bash scripts/ingestion/run_ingest.sh
bash scripts/indexing/run_index.sh
bash scripts/indexing/run_validate_index.sh
```

## 命令行问答

```bash
PYTHONPATH=src python scripts/ask.py
```

启用 LLM 回答：

```bash
PYTHONPATH=src python scripts/ask.py --llm
```

常用参数：

```text
--backend {chroma,elastic,both}
--k N
--mode {strict,balanced,broad}
--no-rerank
--rerank-top-k N
--show-plan
--save-trace
--debug
```

交互命令：

```text
help   显示帮助
debug  切换诊断输出
exit   退出
```

## HTTP 服务

```bash
PYTHONPATH=src python scripts/serve.py
```

默认接口包括：

```text
GET  /healthz
POST /api/ask
POST /api/chat
GET  /api/conversations
GET  /api/conversations/{conversation_id}
```

示例请求：

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "比较蒸汽气化与 CO2 气化对 H2/CO 比的影响",
    "backend": "elastic",
    "k": 10,
    "rerank": true,
    "llm": true,
    "debug": false
  }'
```

## 结构化实验记录

```bash
PYTHONPATH=src python scripts/extract_records.py
PYTHONPATH=src python scripts/export_records.py
```

结构化记录包含工艺阶段、煤种、反应器、温度、压力、气化剂、比例、污染物或产物数值，以及对应证据来源。

## 评估

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --dataset data/eval/complex_science_sample.jsonl \
  --output-dir data/artifacts/evaluation
```

同时评估回答和引用：

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --dataset data/eval/complex_science_sample.jsonl \
  --output-dir data/artifacts/evaluation \
  --answers
```

验证复杂问答数据格式和类型覆盖：

```bash
PYTHONPATH=src python scripts/validate_complex_dataset.py --require-all-types
```

评估输出包括：

```text
metrics.json
case_results.jsonl
failures.jsonl
manifest.json
summary.md
```

## 仓库 Harness

安装 Harness 依赖并执行完整离线验收：

```bash
python -m pip install -r requirements/ci.txt
bash scripts/quality/check_repository.sh
```

Harness 会依次检查：

- Python 编译
- 仓库内部 import 是否有效
- `pyproject.toml` 与 CI 依赖是否一致
- 已安装依赖是否冲突
- Ruff
- mypy
- 全量 pytest

Harness 设置离线环境变量，不下载模型、不调用远程 API，也不要求 Elasticsearch 服务在线。

## 项目结构

```text
src/coal_kb/
├── core/            # 核心模型和接口契约
├── infra/           # 配置、Provider、持久化和可观测性
├── ingestion/       # Loader、解析、Chunking 和元数据
├── tokenization/    # Token 计数与 Provider
├── indexing/        # 索引构建与校验
├── recall/          # Dense、Sparse、父子召回和 RRF
├── reranking/       # 重排序服务
├── retrieval/       # 查询规划、约束和检索编排
├── complex_qa/      # 比较、多跳、聚合、表格和跨文档执行
├── context/         # 证据预算、引用和上下文构建
├── answering/       # 回答、Claim 和置信度
├── application/     # 单轮问答和多轮会话用例
├── interfaces/      # CLI、HTTP API 和前端资源
├── evaluation/      # 数据模型、指标、Pipeline 和报告
├── conversation/    # 会话历史与存储
├── records/         # 结构化记录抽取流程
├── schema/          # 记录 Schema 和校验
└── utils/           # 跨模块公共工具
```

## 设计约束

- 业务模块只保留一个正式实现，不维护并行兼容路径。
- 远程和本地 Provider 由配置显式选择。
- 统计聚合由 Python 基于结构化记录计算，LLM 只负责解释。
- 回答必须来自当前检索证据；证据不足时应拒答或降低结论强度。
- 引用必须能够追溯到来源文件、页码、章节或 Chunk。
- 离线测试不得下载模型、调用远程服务或依赖本地 Elasticsearch。

## License

Apache License 2.0。详见 `LICENSE`。
