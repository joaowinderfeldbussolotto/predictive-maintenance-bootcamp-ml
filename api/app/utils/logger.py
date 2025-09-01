from loguru import logger
import sys


def configure_logging(level: str = "INFO"):
    logger.remove()
    logger.add(sys.stdout, level=level)
