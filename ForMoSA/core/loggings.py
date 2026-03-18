import logging
from rich.logging import RichHandler
from rich.theme import Theme
from rich.console import Console
import os


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


    Authors: Arthur Vigan and Allan Denis
    '''

    logger = logging.getLogger(f"ForMoSA.{name}") if name else logging.getLogger("ForMoSA")
    logger.setLevel(level.upper())
    logger.propagate = False

    if logger.hasHandlers():
        for hdlr in logger.handlers:
            logger.removeHandler(hdlr)

    theme = Theme({
        "logging.level.debug": "dim",
        "logging.level.info": "green",
        "logging.level.warning": "yellow",
        "logging.level.error": "red",
        "logging.level.critical": "bold red",
    })

    console = Console(theme=theme)
    handler = RichHandler(show_time=False, show_path=False, console=console)
    logger.addHandler(handler)

    if logfile:
        fh = logging.FileHandler(logfile, mode="a")
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
