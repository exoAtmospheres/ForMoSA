from __future__ import annotations

import os
import logging
from pathlib import Path

from ForMoSA.core.errors import ForMoSAError
from ForMoSA.core.loggings import setup_logging
from ForMoSA.config.global_config import ConfigPath


class ForMoSAPaths(object):
    '''
    Analysis path class, handles the paths used in the configuration file.

    Parameters
    ----------
    config_file_path (ConfigPath): Instance of class ConfigPath
    logger       (logging.Logger): Logger
    log_level               (str): Level of the Logger

    Authors: Allan Denis
    '''

    def __init__(self, config_path: ConfigPath, logger: logging.Logger | None = None, log_level: str = 'info') -> None:

        if not isinstance(config_path, ConfigPath):
            raise ForMoSAError(f'<Wrong type for config_path: {type(config_path)}. Expected a ConfigPath>', logger)

        self._logger = logger or setup_logging(level=log_level)

        self._observation_path = Path(config_path.observation_path).expanduser()
        self._adapt_store_path = Path(config_path.adapt_store_path).expanduser()
        self._result_path = Path(config_path.result_path).expanduser()
        self._model_path = Path(config_path.model_path).expanduser()

        self._path_error = False

        self._validate()

    # ====================
    # Representation
    # ====================

    def __repr__(self):
        return f'<ForMoSAPath, config_file_path={self.config_file_path}, observation_path={self.observation_path}, adapt_store_path={self.adapt_store_path}, result_path={self.result_path}>'

    # ====================
    # Properties
    # ====================

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
        self._validate()

    @property
    def adapt_store_path(self):
        if not self._adapt_store_path.exists():
            self._logger.info(f' Creating {self._adapt_store_path}')
            self._adapt_store_path.mkdir(parents=True, exist_ok=True)
        return self._adapt_store_path

    @adapt_store_path.setter
    def adapt_store_path(self, path: str | os.PathLike):
        self._adapt_store_path = Path(path).expanduser()
        self._validate()

    @property
    def result_path(self):
        return self._result_path

    @result_path.setter
    def result_path(self, path: str | os.PathLike):
        self._result_path = Path(path).expanduser()
        self._validate()

    @property
    def model_path(self):
        return self._model_path

    @model_path.setter
    def model_path(self, path: str | os.PathLike):
        self._model_path = path
        self._validate()

    @property
    def path_error(self):
        return self._path_error

    @property
    def observation(self):
        return self._observation

    @property
    def grid(self):
        return self._grid

    # ====================
    # LMethods
    # ====================

    def _validate(self) -> None:
        '''
        Validation of the paths.

        Authors: Allan Denis
        '''

        self.logger.debug('Check paths')
        for path in [self._observation_path, self._adapt_store_path, self._model_path, self._result_path]:
            if not isinstance(path, (str, os.PathLike)):
                    raise ForMoSAError(f'Wrong type for path: {type(path)}. Expected a str or os.PathLike', self.logger)

        if not self._model_path.exists():
            raise ForMoSAError(f'No Model file. {self._model_path} does not contain any grid model file', self.logger)

        if not self._observation_path.exists():
            raise ForMoSAError(f'No Model file. {self._observation_path} does not contain any observation file', self.logger)

        if not self._result_path.exists():
            self._logger.info(f'    Creating {self._result_path}')
            self._result_path.mkdir(parents=True, exist_ok=True)

        if not self._adapt_store_path.exists():
            self._logger.info(f'    Creating {self._adapt_store_path}')
            self._adapt_store_path.mkdir(parents=True, exist_ok=True)

        self.logger.info('    Paths checked')




