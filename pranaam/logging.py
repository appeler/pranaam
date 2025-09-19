"""Logging configuration for pranaam package."""

import logging


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name, defaults to 'pranaam'

    Returns:
        Configured logger instance
    """
    logger_name = name or "pranaam"
    logger = logging.getLogger(logger_name)

    # Only configure if no handlers exist (avoid duplicate configuration)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
