import logging


def get_logger(
    debug_level: str = "debug", host: str = "localhost", port: int = 8080
) -> logging.RootLogger:
    """
    Return a generic logger to be called by the main process training a given model.
    """
    logger_level = logging.getLevelName(debug_level)
    logger = logging.getLogger(logger_level)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    format = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(format))

    if not logger.hasHandlers():
        logger.addHandler(handler)

    return logger
