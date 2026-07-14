# Application 与 Interfaces 分层

本阶段只调整应用编排和传输接口的归属，不改变 FastAPI URL、请求响应字段、会话持久化或网页资源内容。

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
- `interfaces/web/`：网页静态资源定位和前后端资源边界。

旧入口 `coal_kb.api`、`coal_kb.qa.ask_pipeline` 和 `coal_kb.chat.orchestrator` 继续保留为兼容 facade。网页静态文件的物理位置本阶段保持不变，避免改变现有部署和打包路径。
