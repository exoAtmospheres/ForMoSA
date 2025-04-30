import numpy as np 
import logging
import os
from pathlib import Path
import glob

import ForMoSA  
from ForMoSA.global_params import GlobalParams
from ForMoSA.model import Model
from ForMoSA.AnalysisPath import AnalysisPath

# log
_log = logging.getLogger(__name__)

class Analysis(object):
    '''
    ForMoSA data analysis class
    
    Parameters
    ----------
    config_file_path : str | os.PathLike
        Path to the configuration file
    adapted : bool, optional 
        Whether the model is adapted to the data, by default False. Can be set to True if the model has already been adapted to the data
    log_level : str, optional
        Log level of the handler, by default ``'info'`` for all important informations.
        
    Returns
    -------
    Analysis : ForMoSA.Analysis | None
        An instance of :class:`~ForMoSA.Analysis` initialized based on the configuration file.

        If `config_file_path` is not a properly configured path or if the configuration fill is missing, `None` is returned.
    '''
    
    def __new__(cls, config_file_path: str | os.PathLike, adapted: bool = False, log_level: str = 'info') -> 'Analysis | None' :

        paths = AnalysisPath(config_file_path)
        
        # Check that the files defined in the config file and the config file itself exist
        if paths.path_error == True:   # A path error is raised in tne AnalysisPath class
            return None
        else:
            analysis = super(Analysis, cls).__new__(cls)
            
        # Inits
        analysis._paths = paths
        analysis._model = Model(paths.model_path)
        analysis._adapted = adapted
 
        return analysis
    

    ##################################################
    # Representation
    ##################################################
    
    def __repr__(self):
        return f'<Analysis, config_file={self.paths.config_file_path}>'

    ##################################################
    # Properties
    ##################################################
    
    @property  
    def paths(self):
        return self._paths
    
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
        