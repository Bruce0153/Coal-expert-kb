# Production deployment

本项目只维护一条正式公网部署路径：GitHub → Railway → FastAPI → Railway 私网 Elasticsearch → 持久化 Volume → 远程模型 Provider。

## 1. 部署边界

生产第一版固定为单实例 FastAPI，不启用水平扩容。原因是当前会话数据库、运行配置、摄入线程和任务状态仍以单实例为边界。Elasticsearch 作为独立 Railway Service，仅通过私网访问，不创建公网端口。

FastAPI 使用仓库根目录 `Dockerfile` 构建。`railway.toml` 负责启动命令、`/ready` 健康检查和重启策略；不要在 Railway 控制台再维护一套不同的启动参数。

## 2. FastAPI Service

从 GitHub 仓库创建 Railway Service，保持根目录为仓库根目录。Railway 会自动识别根目录的 `Dockerfile`。

为 FastAPI Service 创建 Volume，并挂载：

```text
/app/data
```

生产配置会将以下运行数据统一放在该目录下：

```text
/app/data/raw_pdfs
/app/data/raw_docs
/app/data/interim
/app/data/artifacts
/app/data/chroma_db
/app/data/expert.db
/app/data/kb.db
/app/data/manifest.json
```

不要把 API Key、管理员 Secret 或 Session Secret 写入仓库或 Volume 配置文件。

## 3. Elasticsearch Service

在同一 Railway Project / Environment 创建名为 `elasticsearch` 的 Service。镜像版本应与仓库当前本地开发版本保持一致。

Elasticsearch 数据 Volume 挂载：

```text
/usr/share/elasticsearch/data
```

该 Service 不生成 Public Domain，不开放公网 9200。FastAPI 通过 Railway 私网访问：

```text
http://elasticsearch.railway.internal:9200
```

如果修改 Elasticsearch Service 名称，同时修改 `COAL_KB_ELASTIC_URL`。

## 4. 必填环境变量

以 `.env.example` 为唯一变量清单。生产至少需要：

```text
DASHSCOPE_API_KEY=<provider-secret>
COAL_KB_CONFIG=configs/prod.yaml
COAL_KB_PUBLIC_MODE=true
COAL_KB_DATA_ROOT=/app/data
COAL_KB_ELASTIC_URL=http://elasticsearch.railway.internal:9200
COAL_KB_SESSION_SECRET=<strong-random-secret>
COAL_KB_ADMIN_SECRET=<different-strong-random-secret>
COAL_KB_ALLOWED_ORIGINS=https://<railway-or-custom-domain>
```

`COAL_KB_SESSION_SECRET` 与 `COAL_KB_ADMIN_SECRET` 必须使用两个不同的高强度随机值。

公网默认只开放：

```text
COAL_KB_PUBLIC_RESEARCH_ROUTES=standard,graph
```

如需开放 Multimodal 或 Agent，应先完成对应成本、超时和资源上限评估，再显式加入白名单。

## 5. 健康检查

服务提供两类健康检查：

- `/health`：liveness，只判断 FastAPI 进程可响应；
- `/ready`：readiness，检查持久化目录、Elasticsearch 和生产所需 Provider Secret，不调用收费 LLM。

Railway 使用 `/ready` 作为部署健康检查。新容器未 ready 时，不应替换当前可用版本。

## 6. 公网安全模型

普通访客：

- 使用签名匿名 Session Cookie；
- 只能访问自己的 Conversation；
- 不能修改 Provider、Base URL、API Key、Backend、Debug 或服务器级运行参数；
- 受 Query 长度、请求频率和进程级并发限制。

管理员：

- 访问 `/admin`；
- 使用 `COAL_KB_ADMIN_SECRET` 登录；
- 登录状态保存在 HttpOnly Cookie；
- 登录后才可修改运行配置、上传/删除文献和执行摄入。

## 7. 首次上线顺序

1. 创建 Elasticsearch Service 和数据 Volume；
2. 创建 FastAPI Service 和 `/app/data` Volume；
3. 填写 `.env.example` 中的生产变量；
4. 确保 FastAPI 与 Elasticsearch 位于同一 Railway Environment；
5. 部署后先检查 `/health`；
6. 再检查 `/ready` 返回 200；
7. 打开 `/admin` 验证管理员登录；
8. 上传一份测试文献并确认摄入完成；
9. 使用普通浏览器和无痕浏览器分别提问，确认会话互相不可见；
10. 最后再绑定正式自定义域名。

## 8. 备份与恢复

需要备份两类持久化数据：

- FastAPI Volume `/app/data`：原始文献、SQLite、manifest、可选 Chroma 数据；
- Elasticsearch Volume：检索索引。

优先使用 Railway Volume 的备份/恢复能力。恢复时先恢复 Elasticsearch，再恢复 FastAPI Volume，最后启动应用并检查 `/ready`。

原始文献与 SQLite 是最重要的恢复源。即使 Elasticsearch 索引不可恢复，也应能够从原始文献重新执行 ingestion 重建索引。

## 9. Release Gate

正式公开域名前必须同时满足：

- Repository Harness 全绿；
- production smoke test 全绿；
- `/health` 与 `/ready` 正常；
- 匿名会话隔离验证通过；
- Admin / Settings 未认证访问被拒绝；
- Provider Override / Debug 公网锁定；
- Rate Limit 和上传限制生效；
- Elasticsearch 无公网入口；
- FastAPI 与 Elasticsearch Volume 重启后数据不丢失；
- Git 历史中不存在生产 Secret；
- 一次真实文献上传、摄入和问答回归成功。
