import numpy as np 
import logging
import os
from pathlib import Path
import glob

import ForMoSA  
from ForMoSA.global_params import GlobalParams
from ForMoSA.model import Model

# log
_log = logging.getLogger(__name__)

class Analysis(object):
    '''
    ForMoSA data analysis class
    
    Parameters
    ----------
    config_file_path : str | os.PathLike
        Path to the configuration file
    adapt : bool, optional 
        Whether to adapt the model to the data. Can be set to False if the model has already been adapted to the data
    log_level : str, optional
        Log level of the handler, by default ``'info'`` for all important informations.
        
    Returns
    -------
    Analysis : ForMoSA.Analysis | None
        An instance of :class:`~ForMoSA.Analysis` initialized based on the configuration file.

        If `config_file_path` is not a properly configured path or if the configuration fill is missing, `None` is returned.
    '''
    
    def __new__(cls, config_file_path: str | os.PathLike, adapted: bool = False, log_level: str = 'info') -> 'Analysis | None' :

        # Check that the config and files exist
        # The command .expanduser().resolve() transforms the path in a full absolute path, removing any '~' in the path
        config_file = Path(config_file_path).expanduser().resolve()
        
        if not config_file.exists():   # Wrong config file name
            _log.error(f'No config file. {config_file_path} is not a valid configuration path.')
            return None
        else:
            global_params = GlobalParams(config_file_path)
            if global_params.paths.path_error == True:   # A path error is raised in tne AnalysisPath class
                return None
            else:
                analysis = super(Analysis, cls).__new__(cls)
            
        # Inits
        analysis._global_params = global_params
        analysis._config_file = config_file
        analysis._model = Model(global_params.paths.model_path)
        analysis._adapted = adapted
 
        return analysis
    

    ##################################################
    # Properties
    ##################################################
    
    @property  
    def global_params(self):
        return self._global_params
    
    @property  
    def config_file(self):
        return self._config_file 
    
    @property 
    def model(self):
        return self._model
    
    @property 
    def adapted(self):
        return self._adapted
    
    ##################################################
    # Methods
    ##################################################
    
    def _adapt(self):
        for inobs, obs in enumerate(sorted(self.global_params.paths.observation_files)):
            #TODO
            return
        
   
# These lines are just for testing purposes. They will be removed for the final version
config = '/Users/allandenis/These/ForMoSA_Main/51_Eri/config_51Eri_b_ExoREM_all_spectro.ini'
model_path = '/Users/allandenis/test.nc'

analysis = Analysis(config)
        