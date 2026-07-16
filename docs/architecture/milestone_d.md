# Milestone D：研究型系统基础

## D0 研究实验框架

研究实验使用 `coal_kb.research.ResearchExperiment` 固定以下信息：

- 实验名称与实验 ID；
- 研究路线；
- 评估数据集；
- Recall 截断值；
- 自定义实验元数据；
- 开始时间、结束时间和运行时长。

指标计算、失败归因和报告继续复用 `EvaluationPipeline`。每次实验输出原有评估文件，并增加 `experiment.json`，不建立第二套指标实现。

运行命令：

```bash
PYTHONPATH=src python scripts/run_research_experiment.py \
  --name graph-baseline \
  --route graph \
  --dataset data/eval/evaluation_sample.jsonl \
  --output-dir data/artifacts/graph_baseline
```

## D1 Graph route

Graph route 在标准路线已经召回的证据集合中构建关系图，不访问额外数据源。

节点是标准 `Document`。边来自以下可解释关系：

- 相同父块；
- 相同来源文档；
- 相同章节；
- 文本桥接词重合。

Graph route 使用原始排名作为基础分数，并根据头部种子证据与其他节点的连接强度进行稳定重排。Trace 保存节点数、边数、种子、边理由、节点得分和最终顺序。

## 运行边界

- 默认路线始终是 `standard`；
- Graph route 不修改原始召回缓存对象；
- Graph route 不引入图数据库；
- Graph route 不执行网络请求、代码或外部工具；
- Context、Answering、引用和查询日志继续使用正式链路。

## 路线参数

单轮 API 和会话 API 支持：

```json
{
  "research_route": "standard"
}
```

当前支持 `standard` 和 `graph`。后续多模态与受控 Agent 路线使用同一协议扩展。
