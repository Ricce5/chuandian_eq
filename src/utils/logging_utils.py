import logging
import os
from pathlib import Path
from typing import Optional, Union


DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _parse_log_level(level: Optional[Union[int, str]]) -> int:
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        normalized = level.strip().upper()
        if normalized.isdigit():
            return int(normalized)
        return getattr(logging, normalized, logging.INFO)
    return logging.INFO


def setup_logging(
    level: Optional[Union[int, str]] = None,
    log_file: Optional[Union[str, Path]] = None,
    force: bool = False,
) -> logging.Logger:
    """Configure root logger once and optionally attach a file handler."""
    root_logger = logging.getLogger()
    log_level = _parse_log_level(level)

    if force:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

    if not root_logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        root_logger.addHandler(stream_handler)
    root_logger.setLevel(log_level)

    if log_file:
        path = Path(log_file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_file_paths = {
            Path(handler.baseFilename).resolve()
            for handler in root_logger.handlers
            if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None)
        }
        if path not in existing_file_paths:
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
            root_logger.addHandler(file_handler)

    return root_logger
