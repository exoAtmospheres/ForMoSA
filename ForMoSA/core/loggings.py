import logging
from rich.logging import RichHandler
from rich.theme import Theme
from rich.console import Console
import os

NO_LOGGING = 100

def setup_logging(level: str ="INFO", logfile: str | os.PathLike = None, name: str = None):
    '''
    Setup the logging

    Parameters
    ----------
    level : str
        Level of the logger ('INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL')
    logfile : str | os.PathLike
        Path of the log
    name : str
        Name of the logger

    Returns
    -------
    logger : logging.Logger
        Logger

    Notes
    -----
    Authors: Arthur Vigan and Allan Denis
    '''

    logger_name = f"ForMoSA.{name}" if name else "ForMoSA"
    logger = logging.getLogger(logger_name)

    # Normalization
    level = level.upper()

    if level in {"OFF", "NONE"}:
        logger.setLevel(NO_LOGGING)
    else:
        logger.setLevel(getattr(logging, level, logging.INFO))

    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    if level in {"OFF", "NONE"}:
        # If no logging, we return the logger earlier
        return logger

    # Theme Rich
    theme = Theme({
        "logging.level.debug": "dim",
        "logging.level.info": "green",
        "logging.level.warning": "yellow",
        "logging.level.error": "red",
        "logging.level.critical": "bold red",
    })

    console = Console(theme=theme)

    # Handler console
    handler = RichHandler(
        show_time=False,
        show_path=False,
        console=console,
        markup=True,
    )
    logger.addHandler(handler)

    # Handler (optional)
    if logfile:
        fh = logging.FileHandler(logfile, mode="a")
        formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
