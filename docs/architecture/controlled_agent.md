# 受控 Agent Planner 与工具执行

受控 Agent 使用四个明确边界：计划协议、Planner、Tool Registry 和 Budgeted Executor。它不是开放式 Agent，不接受模型生成的任意工具名，也没有 Shell、代码执行或开放网络工具。

## 计划协议

`controlled-agent-plan.v1` 规定：

- 第一步必须是 `retrieve`；
- 步骤编号连续；
- `retrieve`、`graph`、`multimodal` 均不可重复；
- 每一步包含动作、原因和经过 Schema 验证的输入；
- 超出最大步数的候选动作被明确标记为 `truncated`。

Planner 在初始检索后检查问题类型和已有证据模态，生成完整计划。计划生成与执行分离，因此可以独立测试、记录和回放。

## Tool Registry

默认注册表只包含：

1. `retrieve`：执行一次标准证据路线；
2. `graph`：对当前证据执行 typed Graph route；
3. `multimodal`：对当前证据执行多模态重排和显式配置的资产检索。

未注册工具、重复注册、未知输入字段和错误执行顺序都会直接失败。默认工具均不接受自由字符串命令，因此无法通过输入注入 Shell 或网络操作。

## Budgeted Executor

执行器在每一步检查：

- 最大调用次数；
- 最大总执行时间；
- 工具是否注册；
- 输入字段是否符合 ToolSpec；
- 前置证据状态是否满足。

每次调用记录动作、原因、输入输出数量、状态、耗时和失败信息。工具异常采取 fail-closed：记录失败后向上抛出，不自动选择未计划工具，也不无限重试。

## 兼容 Trace

`ControlledAgentRoute` 保留旧字段：

- `policy`
- `allowed_actions`
- `max_steps`
- `steps`
- `stop_reason`
- `duration_ms`

并新增：

- `plan`：完整版本化计划；
- `tool_registry`：本次运行允许的工具和输入字段；
- `budget`：调用次数和时间限制；
- `executions`：逐工具完整执行记录。

API、CLI 和 UI 继续使用原有 `research_route=agent`，不需要修改请求协议。
