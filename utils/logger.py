import logging
import os
import sys
from pathlib import Path

_LOGGERS = {}
_ROOT_CONFIGURED = False


def _ensure_root(log_file: str = "logs/run.log") -> None:
    global _ROOT_CONFIGURED
    if _ROOT_CONFIGURED:
        return
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger("dctm")
    root.setLevel(logging.INFO)
    root.handlers.clear()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    root.propagate = False
    _ROOT_CONFIGURED = True


def get_logger(name: str, log_file: str = "logs/run.log") -> logging.Logger:
    _ensure_root(log_file)
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(f"dctm.{name}")
    _LOGGERS[name] = logger
    return logger
