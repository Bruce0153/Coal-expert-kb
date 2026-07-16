# Milestone D：研究型系统基础

## D0 研究实验框架

`coal_kb.research.ResearchExperiment` 固定实验名称、实验 ID、研究路线、评估数据集、Recall 截断值、自定义元数据和运行时间。指标计算、失败归因和报告继续复用 `EvaluationPipeline`，每次实验增加 `experiment.json`，不维护第二套指标实现。

```bash
PYTHONPATH=src python scripts/run_research_experiment.py \
  --name agent-baseline \
  --route agent \
  --dataset data/eval/evaluation_sample.jsonl \
  --output-dir data/artifacts/agent_baseline
```

支持路线：`standard`、`graph`、`multimodal`、`agent`。

## D1 Graph route

Graph route 在标准路线已经召回的证据集合中构建关系图，不访问额外数据源。

节点是标准 `Document`。边关系包括：

- 相同父块；
- 相同来源文档；
- 相同章节；
- 文本桥接词重合。

原始排名提供基础分数，头部种子与其他节点的连接强度提供关系增益。Trace 保存节点数、边数、种子、边理由、节点得分和最终顺序。

## D2 多模态 route

多模态路线统一处理三类证据：

- `text`：正文、摘要和普通章节；
- `table`：具有 `table_id`、表格章节或表格内容标记的证据；
- `image`：具有 `figure_id`、`image_path`、图像章节或图注格式的证据。

路线根据问题中的图像、曲线、表格等意图对对应模态加权，同时保留全部基础证据。返回文档使用副本追加 `research_modality` 和 `multimodal_score`，不会修改召回缓存对象。

基础版本只组织已经完成文本化、图注化或结构化的多模态证据。它不会在运行时擅自下载视觉模型，也不会对未配置图像执行隐式 OCR。

## D3 受控 Agent route

受控 Agent 只允许以下白名单动作：

1. `retrieve`：执行一次标准路线；
2. `graph`：对当前证据做关系重排；
3. `multimodal`：对当前证据做模态标记和重排。

策略根据问题是否涉及比较、机制、关系、图像或表格决定后续动作。默认最大三步，不接受模型生成的任意工具名，不支持 Shell、代码执行、开放网络请求或无限循环。

Agent Trace 包含：

- 策略版本；
- 允许动作；
- 最大步数；
- 每步动作、原因、输入输出数量、状态和耗时；
- 停止原因；
- 最终路线 Trace。

## 统一接入

单轮 API、会话 API、交互式 CLI 和研究实验命令都使用相同的 `research_route` 参数：

```json
{
  "research_route": "multimodal"
}
```

网页设置中提供四条路线选择，选择结果只影响后续问答请求。默认路线始终是 `standard`。

## 运行边界

- 所有路线返回标准 `Document`；
- Planner、Context、Answering、引用和查询日志继续使用正式链路；
- Graph 与多模态路线只处理标准路线已经召回的证据；
- Agent 只能组合正式路线，不能创建任意工具调用；
- 所有研究路线均可通过 D0 实验框架独立评估和比较。
