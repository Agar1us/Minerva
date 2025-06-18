import logging
import sys
import os

# --- Default Configuration ---
LOG_FORMAT = '%(asctime)s - %(name)s - [%(levelname)s] - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
DEFAULT_LOG_LEVEL = logging.INFO

_CONFIGURED_LOGGERS = set()

def get_logger(
    name: str,
    log_file_path: str = os.getenv('outputs_tinkoff_short'),
    file_mode: str = 'a'  # 'a' for append, 'w' for overwrite
) -> logging.Logger:
    """
    Retrieves and configures a logger instance.

    If called for the first time with a given name, it configures the logger
    with a console handler and optionally a file handler.
    Subsequent calls for the same logger name will return the existing logger,
    but its level (and its handlers' levels) can be updated if a new 'level'
    argument is provided.

    :param name: Name of the logger, typically the caller's ``__name__``.
    :param log_file_path: Path to the log file. If None or empty, no file logging.
                          Defaults to the value of the 'outputs_tinkoff_short' environment variable.
    :param file_mode: File opening mode ('a' for append, 'w' for overwrite).
    :return: Configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    effective_level = DEFAULT_LOG_LEVEL
    logger.setLevel(effective_level)

    if name not in _CONFIGURED_LOGGERS:
        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(effective_level)
        logger.addHandler(console_handler)

        if log_file_path:
            try:
                log_dir = os.path.dirname(log_file_path)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                    
                file_handler = logging.FileHandler(log_file_path, mode=file_mode, encoding='utf-8')
                file_handler.setFormatter(formatter)
                file_handler.setLevel(effective_level)
                logger.addHandler(file_handler)
            except Exception as e:
                sys.stderr.write(
                    f"[LOGGING_CONFIG_ERROR] Failed to create file handler for '{log_file_path}' "
                    f"for logger '{name}': {e}\n"
                )
        
        logger.propagate = False
        
        _CONFIGURED_LOGGERS.add(name)
    else:
        # Logger was already configured by this function.
        # Update levels of its existing handlers to the new effective_level.
        for handler in logger.handlers:
            handler.setLevel(effective_level)
            
    return logger