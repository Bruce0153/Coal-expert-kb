# Public deployment security boundary

This branch introduces the minimum server-side boundary required before Coal Expert KB is exposed to untrusted public traffic.

- Anonymous browser sessions are signed and isolated at the conversation store.
- Admin and runtime-setting writes require an authenticated admin cookie in public mode.
- Public Ask/Chat requests cannot override model providers, base URLs, API keys, retrieval backend, rerank policy, or debug mode.
- Public Ask/Chat traffic is protected by per-session rate limits and a global single-process concurrency limit.
- Admin uploads enforce file-count, extension, per-file size, and total-size limits.
- Development mode remains backward compatible and keeps existing direct configuration workflows.

The production deployment configuration, container entrypoint, persistent storage paths, readiness checks, and Railway service layout are added separately so each stage can be validated independently.
