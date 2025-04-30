import numpy as np 
import logging
import os
from pathlib import Path
import glob
from astropy.io import fits

import ForMoSA  
from ForMoSA.global_params import GlobalParams
from ForMoSA.model import Model
from ForMoSA.AnalysisPath import AnalysisPath

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
        paths = AnalysisPath(config_file_path)
        
        # Check that the files defined in the configuration file and the configuration file itself exist
        if paths.path_error == True:   # A path error is raised in tne AnalysisPath class
            return None
        else:
            analysis = super(Analysis, cls).__new__(cls)
            
        # Inits
        analysis._paths = paths
        analysis._model = Model(paths.model_path)
        analysis._adapted = adapted
        
        # Logging
        logger = logging.getLogger(str(paths.config_file_path))
        logger.setLevel(log_level.upper())
        if logger.hasHandlers():
            for hdlr in logger.handlers:
                logger.removeHandler(hdlr)

        handler = logging.FileHandler(paths._result_path / 'analysis.log', mode='w', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s\t%(levelname)8s\t%(message)s')
        formatter.default_msec_format = '%s.%03d'
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        analysis._logger = logger
        
        analysis._obs_file = dict()
        analysis._obs_name = dict()
        analysis._obs_data = dict()
 
        return analysis
    

    ##################################################
    # Representation
    ##################################################
    
    def __repr__(self) -> str:
        return f'<Analysis, config_file={self.paths.config_file_path}, model={self.model.name}>'

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
    def model(self):
        return self._model
    
    @property 
    def adapted(self):
        return self._adapted
    
    @property  
    def obs_file(self):
        return self._obs_file
    
    @property  
    def obs_name(self):
        _obs_name = dict()
        for indobs, obs in self.obs_file.items():
            _obs_name[indobs] = os.path.splitext(os.path.basename(obs))[0]
        return _obs_name
    
    @property 
    def obs_data(self):
        _obs_data = dict()
        for indobs in self.obs_file.keys():
            _obs_data[indobs] = self.extract_observation(indobs)
        return _obs_data
        
    ##################################################
    # Methods
    ##################################################
    
    def extract_observation(self, indobs):
        """
        Extract the information from the observation file, including the wavelengths (um - vacuum), flux (W.m-2.um.1), errors (W.m-2.um.1), covariance (W.m-2.um.1)**2, spectral resolution, 
        instrument/filter name, transmission (Atmo+inst) and star flux (W.m-2.um.1). The wavelength range is define by the parameter "wav_for_adapt".

        Args:
            global_params  (object): Class containing each parameter
            obs_name          (str): Name of the current observation looping
            indobs            (int): Index of the current observation looping

        Returns:
            - obs_dict       (dict): Dictionay containing all the observationnal entries (photometry, spectroscopy and/or optional)

        Author: Simon Petrus, Matthieu Ravet and Allan Denis
        """
        # Extraction
        with fits.open(self.obs_file[indobs]) as hdul:

            # Check the format of the file and extract data accordingly
            wav = hdul[1].data['WAV']
            flx = hdul[1].data['FLX']
            res = hdul[1].data['RES']
            ins = hdul[1].data['INS']
            try: # Check for spectral covariances
                err = hdul[1].data['ERR']
                cov = np.asarray([]) # Create an empty covariance matrix if not already present in the data (to not slow the inversion)
            except:
                cov = hdul[1].data['COV']
                err = np.sqrt(np.diag(np.abs(cov)))
            try: # Check for transmission
                transm = hdul[1].data['TRANSM']
            except:
                transm = np.asarray([])
            try: # Check for star flux
                star_flx = hdul[1].data['STAR_FLX1'][:,np.newaxis]
                is_star = True
            except:
                star_flx = np.asarray([])   
                is_star = False
            if is_star:
                i = 2
                while True: # In case there is multiple star flux (usually shifted to account for the PSF)
                    try:
                        star_flx = np.concatenate((star_flx, hdul[1].data['STAR_FLX' + str(i)][:,np.newaxis]),axis=1)
                        i += 1
                    except:
                        break
            try:
                is_system = True
                system = hdul[1].data['SYSTEMATICS1'][:,np.newaxis]
            except:
                is_system = False
                system = np.asarray([])
            if is_system:
                i = 2
                while True: # In case there is multiple systematics
                    try:
                        system = np.concatenate((system, hdul[1].data['SYSTEMATICS' + str(i)][:,np.newaxis]),axis=1)
                        i += 1
                    except:
                        break

            # Only take the covariance if you use the chi2_covariance likelihood function (will need to be change when new likelihood functions using the
            # covariance matrix will come)
            #if global_params.logL_type[indobs % len(global_params.logL_type)] != 'chi2_covariance' and global_params.logL_type[indobs % len(global_params.logL_type)] != 'chi2_noisescaling_covariance':
            #    cov = np.asarray([])

            # Filter the NaN and inf values
            nan_mod_ind = (~np.isnan(flx)) & (~np.isnan(err)) & (np.isfinite(flx)) & (np.isfinite(err))
            if len(cov) != 0:
                nan_mod_ind = (nan_mod_ind) & np.all(~np.isnan(cov), axis=0) & np.all(~np.isnan(cov), axis=1) & np.all(np.isfinite(cov), axis=0) & np.all(np.isfinite(cov), axis=1)
            if len(transm) != 0:
                nan_mod_ind = (nan_mod_ind) & (~np.isnan(transm)) & (np.isfinite(transm))
            if len(star_flx) != 0:
                for i in range(len(star_flx[0])):
                    nan_mod_ind = (nan_mod_ind) & (~np.isnan(star_flx.T[i])) & (np.isfinite(star_flx.T[i]))
            if len(system) != 0:
                for i in range(len(system[0])):
                    nan_mod_ind = (nan_mod_ind) & (~np.isnan(system.T[i])) & (np.isfinite(system.T[i])) 
            wav = wav[nan_mod_ind]
            flx = flx[nan_mod_ind]
            res = res[nan_mod_ind]
            ins = ins[nan_mod_ind]
            err = err[nan_mod_ind]
            if len(cov) != 0:
                cov = np.transpose(np.transpose(cov[nan_mod_ind])[nan_mod_ind])
                inv_cov = np.linalg.inv(cov) # Save only the inverse covariance to speed up the inversion
            else:
                inv_cov = np.asarray([])
            if len(transm) != 0 and len(star_flx) != 0:
                transm = transm[nan_mod_ind]
            if len(star_flx) != 0:
                star_flx = np.delete(star_flx, np.where(~nan_mod_ind), axis=0)
            if len(system) != 0:
                system = np.delete(system, np.where(~nan_mod_ind), axis=0)

            # - - - - - - - - - 

            # Separate photometry from spectroscopy
            mask_photo = (res == 0.0)
            # Observation dictionary
            obs_dict = {'wav_photo': wav[mask_photo], # Photometry part
                        'flx_photo': flx[mask_photo],
                        'err_photo': err[mask_photo],
                        'ins_photo': ins[mask_photo],
                        'wav_spectro': wav[~mask_photo], # Spectroscopy part
                        'flx_spectro': flx[~mask_photo],
                        'err_spectro': err[~mask_photo],
                        'res_spectro': res[~mask_photo],
                        'inv_cov': inv_cov, # Optional part
                        'transm': transm,
                        'star_flx': star_flx,
                        'system': system
                        }
            
            return obs_dict
    
    def read_obs(self, global_params):
        '''
        Method to read and store the observation files

        Parameters
        ----------
        global_params  (object): Class containing each parameter
        '''
        for indobs, obs in enumerate(sorted(global_params.paths.observation_files)):
            self._obs_file[indobs] = obs
        
    
    def adapt(self, global_params):
        for inobs, obs in enumerate(sorted(global_params.paths.observation_files)):
            self._logger.info(f'Read observation file {obs}')
            
        
   
# These lines are just for testing purposes. They will be removed for the final version
config = '/Users/allandenis/These/ForMoSA_Main/51_Eri/config_51Eri_b_ExoREM_all_spectro.ini'
model_path = '/Users/allandenis/test.nc'

analysis = Analysis(config)
global_params = GlobalParams(config)
global_params.read_info()
analysis.read_obs(global_params)