from __future__ import annotations

import logging
import os
import glob
from configobj import ConfigObj

from pathlib import Path
import colorlog

from ForMoSA.observation import Observation
from ForMoSA.model_grid import ModelGrid
from ForMoSA.error import ForMoSAError


class ForMoSAPaths(object):
    '''
    Analysis path class, handles the paths used in the configuration file.

    Parameters
    ----------
    config_file_path : str | os.PathLike
        Path to the configuration file.
    logger : Logger used
    '''
    _logger_initialized = False  # Classe-level variable to prevent re-initialization

    def __init__(self, config_file_path: str | os.PathLike, log_level: str = 'info') -> None:
        config = ConfigObj(str(config_file_path), encoding='utf8')

        self._config_file_path = Path(config_file_path).expanduser()
        self._observation_path = Path(config['config_path']['observation_path']).expanduser()
        self._adapt_store_path = Path(config['config_path']['adapt_store_path']).expanduser()
        self._result_path = Path(config['config_path']['result_path']).expanduser()
        self._model_path = Path(config['config_path']['model_path']).expanduser()

        logger = logging.getLogger("ForMoSA")
        logger.setLevel(log_level.upper())
        logger.propagate = False

        # Prevent multiple handler addition
        if not ForMoSAPaths._logger_initialized:
            # Clear any existing handlers once
            while logger.hasHandlers():
                logger.removeHandler(logger.handlers[0])

            # File handler (no color)
            file_handler = logging.FileHandler(self.result_path / 'analysis.log', mode='w', encoding='utf-8')
            file_formatter = logging.Formatter('%(asctime)s\t%(levelname)8s\t%(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

            # Console handler (with color)
            console_handler = colorlog.StreamHandler()
            console_formatter = colorlog.ColoredFormatter(
                fmt='%(log_color)s[%(levelname)s] %(message)s',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'bold_red',
                }
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

            ForMoSAPaths._logger_initialized = True  # Mark as initialized

        self._logger = logger
        self._observation = Observation(self.observation_path, self.logger)
        self._grid = ModelGrid(self.model_path, self.logger)
        self._grid._read_grid()
        self.grid.grid.attrs = self.grid.attrs

        self._path_error = False

    ##################################################
    # Representation
    ##################################################

    def __repr__(self):
        return f'<ForMoSAPath, config_file_path={self.config_file_path}, observation_path={self.observation_path}, adapt_store_path={self.adapt_store_path}, result_path={self.result_path}>'

    def __format__(self) -> str:
        return self.__repr__()

    ##################################################
    # Properties
    ##################################################

    @property
    def logger(self):
        return self._logger

    @property
    def config_file_path(self):
        if not self._config_file_path.exists():
            self._logger.error(f' No config file. {self._config_file_path} is not a valid configuration path.')
            self._path_error = True
            return ''
        else:
            return self._config_file_path

    @property
    def observation_path(self):
        return self._observation_path

    @observation_path.setter
    def observation_path(self, path: str | os.PathLike):
        self._observation_path = Path(path).expanduser()
        self._observation = Observation(self.observation_path, self.logger)

    @property
    def adapt_store_path(self):
        if not self._adapt_store_path.exists():
            self._logger.info(f' Creating {self._adapt_store_path}')
            self._adapt_store_path.mkdir(parents=True, exist_ok=True)
        return self._adapt_store_path

    @adapt_store_path.setter
    def adapt_store_path(self, path: str | os.PathLike):
        self._adapt_store_path = Path(path).expanduser()

    @property
    def result_path(self):
        if not self._result_path.exists():
            self._result_path.mkdir(parents=True, exist_ok=True)
        return self._result_path

    @result_path.setter
    def result_path(self, path: str | os.PathLike):
        self._result_path = Path(path).expanduser()

    @property
    def model_root(self):
        return self._model_path.parent

    @property
    def model_path(self):
        if not self._model_path.exists():
            self._logger.error(f' No Model file. {self.model_root} does not contain any grid model file.')
            self._path_error = True
            return ''
        return self._model_path

    @model_path.setter
    def model_path(self, path: str | os.PathLike):
        self._model_path = Path(path).expanduser()
        self._grid = ModelGrid(self.model_path, self.logger)
        self._grid._read_grid()
        self.grid.grid.attrs = self.grid.attrs

    @property
    def observation_files(self):
        files = [f for f in glob.glob(str(self.observation_path)) if f.lower().endswith('.fits')]
        if len(files) == 0:  # No observation
            self._logger.error(f' No observation. {self.observation_path} does not contain any observation.')
            return ForMoSAError()
        else:
            return files

    @property
    def path_error(self):
        return self._path_error

    @property
    def observation(self):
        return self._observation

    @property
    def grid(self):
        return self._grid




