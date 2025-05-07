import numpy as np 
import logging
import os
from pathlib import Path
import glob
from astropy.io import fits

import ForMoSA  
from ForMoSA.global_params import GlobalParams
from ForMoSA.ModelGrid import ModelGrid
from ForMoSA.ForMoSAPaths import ForMoSAPaths
from ForMoSA.observation import Observation
from scipy.interpolate import interp1d
from ForMoSA.utils_spec import resolution_decreasing, continuum_estimate
import ForMoSA.utils as utils
import colorlog

# log
_log = logging.getLogger(__name__)

class ForMoSAError(Exception):
    pass

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

        # PathAnalysis method handling the paths used in the configuration file
        paths = ForMoSAPaths(config_file_path)
        
        # Check that the files defined in the configuration file and the configuration file itself exist
        if paths.path_error == True:   # A path error is raised in tne AnalysisPath class
            return None
        else:
            analysis = super(Analysis, cls).__new__(cls)
            
        logger = logging.getLogger("ForMoSA")

        if logger.hasHandlers():
            logger.handlers.clear()

        logger.setLevel(log_level.upper())
        
        # File handler (no color)
        file_handler = logging.FileHandler(paths._result_path / 'analysis.log', mode='w', encoding='utf-8')
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

        analysis._logger = logger
        
        # Inits
        analysis._paths = paths
        analysis._observation = Observation(paths.observation_path, logger)
        analysis._grid = ModelGrid(paths.model_path, logger)
        analysis._adapted = adapted  

 
        return analysis
    

    ##################################################
    # Representation
    ##################################################
    
    def __repr__(self) -> str:
        return f'<Analysis, config_file={self.paths.config_file_path}, grid={self.grid.name}>'

    def __format__(self) -> str:
        return self.__repr__()

    ##################################################
    # Properties
    ##################################################
    
    @property
    def loglevel(self) -> str:
        return logging.getLevelName(self._logger.level)

    @loglevel.setter
    def loglevel(self, level: str):
        self._logger.setLevel(level.upper())
    
    @property  
    def paths(self):
        return self._paths
    
    @property  
    def config_file(self):
        return self._config_file 
    
    @property 
    def grid(self):
        return self._grid
    
    @property 
    def observation(self):
        return self._observation
    
    @property 
    def adapted(self):
        return self._adapted
    
        
    ##################################################
    # Methods
    ##################################################
    
    
    def adapt(self, observation_files: list, target_res_obs: list=['obs'], target_res_mod: list=['obs'], res_cont: list=['NA'], wav_cont: list=[], hc_type: list=['NA'], rv: list=['NA'], emulator: str='NA', interp_method: list=['linear']):
        '''
        Method to adapt the grid of model to each observation

        Args
            observation_files (list): List of the observation files
            target_res_obs    (list): Target resolution of the observation
            res_cont          (list): Resolution of the continuum
            wav_cont          (list): Wavelength range on which we want to estimate the continuum
            hc_type           (list): High contrast function
            rv                (list): Prior to put in the rv (will be probably removed later)
            emulator          (list): Method of adaptation of the grid ('NA', 'PCA' of 'NMF')
        '''
        
        # Check that inputs are of type 'list'
        is_not_list = utils.check_format(observation_files, target_res_obs, res_cont, hc_type, type_expected=list)
        if len(is_not_list) > 0:
                msg = f" Params in wrong format : {', '.join(is_not_list)}"
                self._logger.critical(msg)
                raise ForMoSAError(msg)
        
        for indobs, obs in enumerate(observation_files):
            
            obs_data_spectro = self.observation.obs_data[indobs]['spectroscopy']
            obs_data_photo = self.observation.obs_data[indobs]['photometry']
            obs_name = self.observation.obs_name[indobs]
            
            self._logger.info(f' Current observation: {obs_name}')
            
            # Adapt observation in case you need to degrade the resolution of the observations
            self._logger.debug(f'Adapt observation {obs_name}')
   
            # Setup target resolution for the observation
            # Interpolate the resolution of the model onto the wavelength of the data to properly decrease the resolution if necessary
            interp_mod_to_obs = interp1d(self.grid.wavelength, self.grid.resolution, fill_value='extrapolate')
            res_mod_obs = interp_mod_to_obs(obs_data_spectro['wav'])

            if target_res_obs[indobs % len(target_res_obs)] == 'obs': # Keeping the resolution of the observation except where its higher than the model's
                _target_res_obs = np.minimum(obs_data_spectro['res'], res_mod_obs)
            else:                                             # Using a custom resolution except where its higher than the model's or the observation's
                res_custom = np.full(len(obs_data_spectro['res']), float(target_res_obs[indobs % len(target_res_obs)]))
                _target_res_obs = np.minimum(obs_data_spectro['res'], res_mod_obs, res_custom)

            if len(_target_res_obs) > 0:
                self.observation._adapt_observation(_target_res_obs, res_cont[indobs % len(res_cont)], wav_cont[indobs % len(wav_cont)], hc_type[indobs % len(hc_type)], indobs)
                self._logger.info(f' Observation {obs_name} adapted.')
            
            # Save the data
            self._logger.debug(f'> Save observation file {self.paths.result_path}' + f'/spectrum_obs_{obs_name}.npz')
            np.savez(os.path.join(self.paths.result_path, f'spectrum_obs_{obs_name}.npz'), **self.observation.obs_data[indobs])
            
            if not self.adapted:   # If the model is not already adapted to the data, or if the user wants to redo the adaptation
                wavelength_photo, ins_photo = obs_data_photo['wav'], obs_data_photo['ins']
                # Setup target wavelength and resolution for the observation and the model
                if target_res_mod[indobs % len(target_res_mod)] == 'mod': # Kepping the model's resolution
                    target_wavelength, target_resolution = self.grid.wavelength, self.grid.resolution
                if target_res_mod[indobs % len(target_res_mod)] == 'obs': # Using the observation's resolution except where its higher than the model's
                    target_wavelength, target_resolution = obs_data_spectro['wav'], obs_data_spectro['res']
                else:                                             # Using a custom resolution except where its higher than the model's
                    target_wavelength, target_resolution = self.grid.wavelength, np.full(len(self.grid.wavelength), float(target_res_mod[indobs % len(target_res_mod)]))
   
                if len(obs_data_spectro['wav']) > 0:
                    # # Masks to have larger cuts of the spectroscopic grid if needed (if rv is defined)
                    if rv[indobs*3 % len(rv)] == 'NA':
                        mask_mod_obs = (target_wavelength <= obs_data_spectro['wav'][-1]) & (target_wavelength >= obs_data_spectro['wav'][0]) 
                        target_wavelength, target_resolution = target_wavelength[mask_mod_obs], target_resolution[mask_mod_obs]
                    else:
                        mask_mod_obs = (target_wavelength <= 1.01 * obs_data_spectro['wav'][-1]) & (target_wavelength >= 0.99 * obs_data_spectro['wav'][0])   # 1.01 corresponds to a value of 3000 km/s for the RV so we do no risk to lose data on the edges when applying the RV correction
                        target_wavelength, target_resolution = target_wavelength[mask_mod_obs], target_resolution[mask_mod_obs]

                self.grid.adapt_grid(target_resolution, target_wavelength, wavelength_photo, ins_photo, wav_cont, res_cont, False, obs_name)

                if emulator == 'PCA':
                    self._logger.info(' Decomposing the grid using PCA')
                    
                    # TODO
                    
                if emulator == 'NMF':
                    self._logger.info(' Decomposing the grid using NMF')
                    
                    # TODO
                    
        self.grid._interpolate_missing_values()
        self.grid._save_grid(self.paths.adapt_store_path)
        
                
        
   
# These lines are just for testing purposes. They will be removed for the final version
config = '/Users/allandenis/These/ForMoSA_Main/51_Eri/config_51Eri_b_ExoREM_all_spectro.ini'
model_path = '/Users/allandenis/test.nc'

analysis = Analysis(config, log_level='debug')
global_params = GlobalParams(config)
analysis.adapt(observation_files=global_params.paths.observation_files, target_res_obs=global_params.target_res_obs, res_cont=global_params.res_cont, wav_cont=global_params.wav_cont, hc_type=global_params.hc_type, interp_method=global_params.method)