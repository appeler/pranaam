"""Logging configuration for pranaam package."""

import logging

from rich.logging import RichHandler


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name, defaults to 'pranaam'

    Returns:
        logging.Logger: Configured logger instance
    """
    logger_name = name or "pranaam"
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        handler = RichHandler(
            show_time=True, show_path=False, rich_tracebacks=True, markup=True
        )
        formatter = logging.Formatter("%(name)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
