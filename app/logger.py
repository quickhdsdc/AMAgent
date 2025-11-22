# app/logger.py
import sys
import logging
from datetime import datetime
from pathlib import Path
from loguru import logger as _logger

from app.config import PROJECT_ROOT

_print_level = "INFO"


class InterceptHandler(logging.Handler):
    """Redirect stdlib logging records into Loguru."""

    def emit(self, record: logging.LogRecord):
        try:
            level = _logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        _logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def define_log_level(print_level="INFO", logfile_level="DEBUG", name: str = None):
    """
    Adjust the log level and configure Loguru sinks.
    Also installs a stdlib -> Loguru bridge so all logs end up in Loguru.
    """
    global _print_level
    _print_level = print_level

    # --- stdlib -> Loguru bridge ---
    logging.root.handlers = []
    logging.root.setLevel(logging.NOTSET)
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # --- Loguru sinks ---
    current_date = datetime.now().strftime("%Y%m%d%H%M%S")
    log_name = f"{name}_{current_date}" if name else current_date
    log_dir = PROJECT_ROOT / "logs"
    _ensure_dir(log_dir)

    _logger.remove()  # remove default sink
    _logger.add(sys.stderr, level=print_level)
    _logger.add(log_dir / f"{log_name}.log", level=logfile_level, enqueue=True)

    return _logger


# Initialize once on import
logger = define_log_level()


if __name__ == "__main__":
    logger.info("Starting application")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")

