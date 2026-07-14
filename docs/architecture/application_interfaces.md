# Application 与 Interfaces 分层

应用编排和传输接口已经统一到以下 canonical 结构，不再保留重复的 API、Chat、QA 或 Web 模块。

```text
interfaces/api              interfaces/web
      │                           │
      └──────────┬────────────────┘
                 ↓
            application
       ask / chat / admin
                 ↓
retrieval / context / answering / ingestion / persistence
```

职责边界：

- `application/ask.py`：运行时组装、单轮执行、日志与响应字典。
- `application/chat.py`：会话历史、消息持久化和多轮问答编排。
- `application/admin.py`：文档统计、上传、删除与摄入用例。
- `interfaces/api/`：FastAPI 模型、路由、Presenter 和异常映射。
- `interfaces/cli/`：命令行展示组件。
- `interfaces/web/`：网页静态资源和资源路径。

依赖只能从接口层指向应用层。`application` 不得依赖 FastAPI、网页资源或其他接口实现。
