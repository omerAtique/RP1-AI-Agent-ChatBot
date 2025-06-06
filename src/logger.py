import logging

def get_logger(name: str = __name__):
    return logging.getLogger(name)

def log_info(message: str, logger_name: str = "app"):
    get_logger(logger_name).info(message)

def log_warning(message: str, logger_name: str = "app"):
    get_logger(logger_name).warning(message)

def log_error(message: str, exception: Exception = None, logger_name: str = "app"):
    logger = get_logger(logger_name)
    if exception:
        logger.error(f"{message}: {str(exception)}", exc_info=True)
    else:
        logger.error(message)