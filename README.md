# Coal Expert KB

Coal Expert KB 是面向煤热解、气化与燃烧文献的证据驱动知识库。系统将论文、报告、表格和结构化数据转换为可检索证据，通过统一查询规划、检索、研究路线、上下文构建和回答链路提供带引用的科研问答。

## 核心能力

- 摄入 PDF、Markdown、文本、HTML、DOCX、PPTX、CSV、XLSX、JSON 和 JSONL。
- 按标题层级和语义边界构建父子 Chunk，保留来源、页码、章节和父子关系。
- 支持 Chroma、Elasticsearch 和混合检索，以及约束、两阶段检索、RRF、Rerank 和来源多样性控制。
- 支持事实、条件、比较、多跳、统计聚合、表格和跨文档问题。
- 支持 Standard、Graph、Multimodal 和受控 Agent 研究路线。
- 提供运行中参考文献上传、增量入库、任务进度、文档管理和知识库统计。
- 在网页中配置 Tokenizer、Embeddings、Rerank、LLM 的远程或本地 Provider。
- 提供 CLI、HTTP API、网页、会话历史、离线评估和统一仓库 Harness。

## 运行链路

```text
Document
  → Loader / Parser
  → Metadata normalization
  → Hierarchical chunking
  → Chroma / Elasticsearch / SQLite

Question
  → QueryPlanner
  → Standard / Graph / Multimodal / Controlled Agent
  → ContextBuilder
  → Answerer
  → Claims + Citations + Source cards
```

所有问答路线最终返回标准 `Document`，并复用同一套上下文、引用、回答和查询日志。

## 环境与安装

- Python 3.10 或更高版本
- Elasticsearch 8.x，仅在使用 Elasticsearch 后端时需要
- 远程 API Key，仅在对应能力使用 `remote` 模式时需要
- 本地模型或本地兼容服务，仅在对应能力使用 `local` 模式时需要

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

可选依赖：

```bash
python -m pip install -e ".[docs]"
python -m pip install -e ".[training]"
python -m pip install -e ".[dev,docs]"
```

## 配置

主配置文件是 `configs/app.yaml`。也可以通过 `COAL_KB_CONFIG` 指向其他配置文件。

每项模型能力都显式选择 `remote` 或 `local`，不会在调用失败后自动切换实现。

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
    device: auto
    dimensions: 1024

rerank:
  mode: local
  remote:
    provider: dashscope
    base_url: "https://dashscope.aliyuncs.com/api/v1/services/rerank"
    api_key_env: "DASHSCOPE_API_KEY"
    model: "qwen3-rerank"
  local:
    provider: cross_encoder
    model: "BAAI/bge-reranker-base"
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
```

```bash
export DASHSCOPE_API_KEY="..."
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
├── artifacts/      # 模型、评估和研究实验产物
└── eval/           # 评估数据集

storage/
├── chroma_db/      # Chroma 数据
├── expert.db       # 结构化实验记录
└── kb.db           # 注册库、查询日志和会话
```

## 文档摄入与索引

```bash
PYTHONPATH=src python scripts/ingest.py
PYTHONPATH=src python scripts/index.py
PYTHONPATH=src python scripts/validate_index.py
```

```bash
PYTHONPATH=src python scripts/ingest.py --rebuild
PYTHONPATH=src python scripts/ingest.py --tables --table-flavor auto
```

## 网页与运行中增量入库

启动服务：

```bash
PYTHONPATH=src python scripts/serve.py
```

打开 `http://127.0.0.1:8000/`。

“知识库管理”提供：

- 拖拽或选择多个参考文献；
- 文件传输进度；
- 保存、解析、分块、向量化和索引阶段提示；
- 上传后自动增量入库开关；
- 文档列表、删除、统计和手动摄入；
- 单工作线程串行更新索引，避免并发写入。

上传返回任务 ID，前端通过 `/api/admin/tasks/{task_id}` 获取状态。上传过程中问答服务可以继续运行；新文献完成索引后进入后续检索。

## 网页 Provider 设置

“设置”可以配置：

- 检索后端、模式、Top-K 和 Rerank；
- Tokenizer 的模式、Provider、地址、模型和远程 API Key；
- Embeddings 的模式、Provider、地址、模型和远程 API Key；
- Rerank 的模式、Provider、地址、模型和远程 API Key；
- LLM 的模式、Provider、地址、模型和远程 API Key；
- Standard、Graph、Multimodal 或 Agent 研究路线。

设置立即作用于后续问答和增量入库。API Key 不写入配置文件，仅保存在服务进程和当前浏览器会话。切换 Embedding 模型时必须保证与已有索引向量空间一致，否则应重建索引。

## 命令行问答

```bash
PYTHONPATH=src python scripts/ask.py
PYTHONPATH=src python scripts/ask.py --llm
PYTHONPATH=src python scripts/ask.py --research-route graph --debug
```

常用参数：

```text
--backend {chroma,elastic,both}
--k N
--mode {strict,balanced,broad}
--research-route {standard,graph,multimodal,agent}
--no-rerank
--rerank-top-k N
--show-plan
--save-trace
--debug
```

## HTTP API

主要接口：

```text
GET    /health
POST   /api/ask
POST   /api/chat
GET    /api/conversations
GET    /api/settings/runtime
PUT    /api/settings/runtime
DELETE /api/settings/runtime
POST   /api/admin/documents/upload
GET    /api/admin/tasks/{task_id}
GET    /api/admin/documents
DELETE /api/admin/documents/{document_id}
POST   /api/admin/ingest
GET    /api/admin/stats
```

示例：

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "比较图表中的蒸汽气化与 CO2 气化结果",
    "backend": "elastic",
    "k": 10,
    "rerank": true,
    "research_route": "agent",
    "llm": true,
    "debug": true
  }'
```

## Milestone D 研究路线

### Standard

执行正式查询规划和复杂问答路线，不增加研究型重排。

### Graph

在已召回证据间构建相同父块、来源、章节和桥接词关系，并保存节点、边、理由、得分和顺序 Trace。Graph 不访问额外数据源。

### Multimodal

根据元数据和内容统一标记文本、表格和图像/图注证据，并按问题中的图、曲线、表格意图重排。基础版本不隐式下载视觉模型或执行未经配置的 OCR。

### Controlled Agent

只允许 `retrieve`、`graph`、`multimodal` 三种动作，默认最多三步。每步记录动作、原因、输入输出数量、状态和耗时；不支持任意代码、Shell、开放网络请求或无限循环。

## 研究实验

```bash
PYTHONPATH=src python scripts/run_research_experiment.py \
  --name multimodal-baseline \
  --route multimodal \
  --dataset data/eval/complex_science_sample.jsonl \
  --output-dir data/artifacts/multimodal_baseline
```

实验复用统一 `EvaluationPipeline`，输出：

```text
experiment.json
metrics.json
case_results.jsonl
failures.jsonl
manifest.json
summary.md
```

## 常规评估

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --dataset data/eval/complex_science_sample.jsonl \
  --output-dir data/artifacts/evaluation \
  --answers

PYTHONPATH=src python scripts/validate_complex_dataset.py --require-all-types
```

## 仓库 Harness

```bash
python -m pip install -r requirements/ci.txt
bash scripts/quality/check_repository.sh
```

Harness 检查 Python 编译、内部 import、依赖一致性、`pip check`、Ruff、mypy 和全量 pytest。离线验收不下载模型、不调用远程 API，也不要求 Elasticsearch 在线。

## 项目结构

```text
src/coal_kb/
├── core/            # 核心模型和契约
├── infra/           # 配置、Provider、持久化和可观测性
├── ingestion/       # Loader、解析、Chunking 和元数据
├── indexing/        # 索引构建与校验
├── recall/          # Dense、Sparse、父子召回和 RRF
├── retrieval/       # 查询规划、约束和检索编排
├── complex_qa/      # 比较、多跳、聚合、表格和跨文档执行
├── research/        # 实验、Graph、多模态和受控 Agent 路线
├── context/         # 证据预算、引用和上下文构建
├── answering/       # 回答、Claim 和置信度
├── application/     # 问答、会话、管理和运行配置
├── interfaces/      # CLI、HTTP API 和网页
├── evaluation/      # 数据、指标、Pipeline 和报告
├── conversation/    # 会话历史与存储
├── records/         # 结构化记录抽取
└── utils/           # 公共工具
```

## 设计约束

- 每项业务能力只保留一个正式实现，不维护并行兼容路径。
- 远程和本地 Provider 由配置显式选择。
- 回答必须来自当前证据；证据不足时拒答或降低结论强度。
- 引用必须可追溯到来源、页码、章节或 Chunk。
- Graph 和多模态路线只处理标准路线已召回的证据。
- Agent 只能执行固定白名单动作并受最大步数约束。
- 离线测试不得下载模型、调用远程服务或依赖本地 Elasticsearch。

## License

Apache License 2.0。详见 `LICENSE`。
