"""启动 Coal Expert KB 的 FastAPI Web 服务。"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import uvicorn


@dataclass
class Serve:
    host: str
    port: int
    reload: bool

    def process(self) -> None:
        uvicorn.run(
            "coal_kb.interfaces.api.app:app",
            host=self.host,
            port=self.port,
            reload=self.reload,
        )


def _public_mode() -> bool:
    return os.getenv("COAL_KB_PUBLIC_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Coal Expert KB web app.")
    parser.add_argument("--host", default="0.0.0.0" if _public_mode() else "127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    Serve(host=args.host, port=args.port, reload=args.reload).process()


if __name__ == "__main__":
    main()

# 运行命令：python scripts/serve.py
