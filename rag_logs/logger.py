import logging
from logging import StreamHandler
from logging.handlers import RotatingFileHandler


def configure_logging() -> None:
    """
    Configures the root logger for the application:
    - StreamHandler for console output
    - RotatingFileHandler for file output
        - Writes to `rag_logs/logs.log`
        - Maximum file size of 1MB
        - No backup files, rotation discards old logs
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            StreamHandler(),
            RotatingFileHandler(
                filename="rag_logs/logs.log",
                maxBytes=1024*1024,
                backupCount=0
            )
        ]
    )
