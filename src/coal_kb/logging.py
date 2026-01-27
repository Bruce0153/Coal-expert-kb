from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler

from .settings import AppConfig


def setup_logging(cfg: AppConfig, *, logger_name: Optional[str] = None) -> None:
    level = getattr(logging, cfg.logging.level.upper(), logging.INFO)

    handlers = [RichHandler(rich_tracebacks=True, markup=True)]
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )

    if logger_name:
        logging.getLogger(logger_name).setLevel(level)
