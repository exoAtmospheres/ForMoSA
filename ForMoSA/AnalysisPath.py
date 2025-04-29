from __future__ import annotations

# standard libraries
import shutil
import logging
import os
import glob
from configobj import ConfigObj

from pathlib import Path
from typing import Any

# log
_log = logging.getLogger(__name__)

# Format logging for this module
if not _log.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)  # Minimal logging level for this module
    formatter = logging.Formatter('%(asctime)s\t%(levelname)8s\t%(message)s')
    formatter.default_msec_format = '%s.%03d'
    handler.setFormatter(formatter)
    _log.addHandler(handler)

_log.setLevel(logging.INFO)
_log.propagate = False 

class AnalysisPath(object):
    '''
    Analysis path class, handles the paths used in the configuration file.
    
    Parameters
    ----------
    config_file_path : str | os.PathLike
        Path to the configuration file.
    '''
    
    def __init__(self, config_file_path: str | os.PathLike) -> None:
        _log.info('Read configuration file')
        config = ConfigObj(config_file_path, encoding='utf8')

        self._config_file_path = Path(config_file_path).expanduser()
        self._observation_path = Path(config['config_path']['observation_path'] + '*').expanduser()
        self._adapt_store_path = Path(config['config_path']['adapt_store_path']).expanduser()
        self._result_path = Path(config['config_path']['result_path']).expanduser()
        self._model_path = Path(config['config_path']['model_path']).expanduser()
        self._path_error = False
        
    ##################################################
    # Representation
    ##################################################

    def __repr__(self):
        return f'<AnalysisPath, config_file_path={self.config_file_path}, observation_path={self.observation_path}, adapt_store_path={self.adapt_store_path}, result_path={self.result_path}>'

    def __format__(self) -> str:
        return self.__repr__()
    
    ##################################################
    # Properties
    ##################################################
    
    @property  
    def config_file_path(self):
        return self._config_file_path
    
    @property 
    def observation_path(self):
        return self._observation_path
    
    @property 
    def adapt_store_path(self):
        if not self._adapt_store_path.exists():
            _log.info(f'Creating {self._adapt_store_path}')
            os.mkdir(self._adapt_store_path)
        return self._adapt_store_path
    
    @property 
    def result_path(self):
        if not self._result_path.exists():
            _log.info(f'Creating {self._result_path}')
            os.mkdir(self._result_path)
        return self._result_path
    
    @property 
    def model_root(self):
        return self._model_path.parent
    
    @property 
    def model_path(self):
        if not self._model_path.exists():
            _log.error(f' No Model file. {self.model_root} does not contain any grid model file.')
            self._path_error = True
            return ''
        return self._model_path
    
    @property  
    def observation_files(self):
        files = glob.glob(str(self.observation_path))
        if len(files) == 0:  # No observation
            _log.error(f' No observation. {self.observation_path} does not contain any observation.')
            self._path_error = True
            return []
        else:
            return files
        
    @property  
    def path_error(self):
        return self._path_error

    
        
    