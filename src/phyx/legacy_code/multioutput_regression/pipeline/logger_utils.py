import sys

from loguru import logger


def setup_logger(log_file="pipeline.log"):
    logger.remove()  # Remove the default handler
    logger.add(
        log_file,
        level="DEBUG",
        rotation="5 MB",
        retention="10 days",
    )
    logger.add(sys.stderr, level="DEBUG")  # Add console sink (stderr)
