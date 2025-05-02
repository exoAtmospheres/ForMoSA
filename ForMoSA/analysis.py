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
from scipy.interpolate import interp1d
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import multiprocessing as mp
from ForMoSA.utils_spec import resolution_decreasing, continuum_estimate

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
        return self._obs_data
        
    ##################################################
    # Methods
    ##################################################
    
    
    def _islist(self, *params):
        '''
        Method to check that all the components defined in params are a list

        Args
            *params : list of parameters
            
        Author: Allan Denis
        '''
        for param in params:
            if not(isinstance(param, list)):
                self._logger.critical(f' {param} is not a list.')
                raise ForMoSAError()
    
    
    def extract_observation(self, indobs=0):
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
        self._logger.info(f'Extract observation data from {self.obs_file[indobs]}')
        
        with fits.open(self.obs_file[indobs]) as hdul:
            self._logger.debug('> Extract information from file {self.obs_file[indobs]}')

            # Check the format of the file and extract data accordingly
            try:
                wav = hdul[1].data['WAV']
                flx = hdul[1].data['FLX']
                res = hdul[1].data['RES']
                ins = hdul[1].data['INS']
            except KeyError:
                self._logger.critical(' Key error.')
                raise ForMoSAError()
            
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
            
            # Check-ups and warnings for negative values in the diagonal of the covariance matrix
            if len(wav[~mask_photo]) != 0 and any(np.diag(inv_cov) < 0):
                self._logger.critical("Negative value(s) is(are) present on the diagonal of the covariance matrix.") 
                raise ForMoSAError
           
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
    
    
    def adapt_observation(self, obs_data: dict, obs_name: str, target_res_obs: str='obs', res_cont: str='NA', wav_cont: np.ndarray=[], hc_type: str='NA'):
        """
        Decrease the spectral resolution of the current observation and remove the continuum if necessary

        Args:
            obs_data          (dict): Dictionnaire of the current observation
            obs_name           (str): Name of the current observation
            target_res_obs     (str): Target resolution of the observation
            res_cont           (str): Resolution of the continuum
            wav_cont       (ndarray): Wavelength of the continuum
            hc_type            (str): High contrast function
            
        Returns:
            obs_data          (dict): Adapted current observation

        Author: Simon Petrus, Matthieu Ravet and Allan Denis
        """

        self._logger.info(' Adapting observation {self.obs_file[indobs]}')

        # Decrease the resolution and remove the continuum if necessary
        if len(obs_data['wav_spectro']) != 0:

            # - - - - - -
            # Setup target resolution for the observation
            # Interpolate the resolution of the model onto the wavelength of the data to properly decrease the resolution if necessary
            interp_mod_to_obs = interp1d(self.model.wavelength, self.model.resolution, fill_value='extrapolate')
            res_mod_obs = interp_mod_to_obs(obs_data['wav_spectro'])

            if target_res_obs == 'obs': # Keeping the resolution of the observation except where its higher than the model's
                target_res_obs = np.minimum(obs_data['res_spectro'], res_mod_obs)
            else:                                             # Using a custom resolution except where its higher than the model's or the observation's
                res_custom = np.full(len(obs_data['res_spectro']), float(target_res_obs))
                target_res_obs = np.minimum(obs_data['res_spectro'], res_mod_obs, res_custom)

            # - - - - - -

            # If we want to decrease the resolution of the spectroscopic data:
            self._logger.info(' Decreasing the resolution of the data if necessary.')
            obs_data['flx_spectro'] = resolution_decreasing(obs_data['wav_spectro'],
                                                            obs_data['flx_spectro'],
                                                            obs_data['res_spectro'],
                                                            obs_data['wav_spectro'],
                                                            target_res_obs)
            
            obs_data['transm'] = resolution_decreasing(obs_data['wav_spectro'],
                                                       obs_data['transm'],
                                                       obs_data['res_spectro'],
                                                       obs_data['wav_spectro'],
                                                       target_res_obs)
            
            obs_data['star_flx'] = np.asarray([resolution_decreasing(obs_data['wav_spectro'],
                                                                     obs_data['star_flx'][:,i],
                                                                     obs_data['res_spectro'],
                                                                     obs_data['wav_spectro'],
                                                                     target_res_obs) for i in range(obs_data['star_flx'].shape[-1])]).T
        
            obs_data['system'] = np.asarray([resolution_decreasing(obs_data['wav_spectro'],
                                                                   obs_data['system'][:,i],
                                                                   obs_data['res_spectro'],
                                                                   obs_data['wav_spectro'],
                                                                   target_res_obs) for i in range(obs_data['system'].shape[-1])]).T
        
        
            # Since the resolution of the observation might have change, we need to save the new one
            obs_data['res_spectro'] = target_res_obs
            
            # If we want to estimate and substract the continuum of the data:
            if res_cont != 'NA':
                self._logger.info(' Substract the continuum to the data')
                self._logger.info(f' {obs_name} will have a R = {res_cont} continuum removed using a {wav_cont} wavelength range')

                obs_data['flx_spectro_cont'] = continuum_estimate(obs_data['wav_spectro'], 
                                                                  obs_data['flx_spectro'], 
                                                                  obs_data['res_spectro'], 
                                                                  wav_cont, res_cont)
                
                # If you don't use hc models, the data continuum is directly removed
                if hc_type == 'NA':
                    obs_data['flx_spectro'] -= obs_data['flx_spectro_cont']

                else: # If you use hc models, the data is kept; we just need to estimate the continuum of the star flux as well
                    obs_data['star_flx_cont'] = continuum_estimate(obs_data['wav_spectro'],
                                                                obs_data['star_flx'][:,len(obs_data['star_flx'][0]) // 2], # Continuum of the star on the central pixel
                                                                obs_data['res_spectro'],
                                                                wav_cont, res_cont)
                 
        return obs_data
    
    
    def adapt(self, observation_files: list, target_res_obs: list=['obs'], res_cont: list=['NA'], wav_cont: list=[], hc_type: list=['NA']):
        '''
        Method to adapt the grid of model to each observation

        Args
            observation_files (list): List of the observation files
            target_res_obs (list): Target resolution of the observation
            res_cont (list): Resolution of the continuum
            wav_cont (list): Wavelength range on which we want to estimate the continuum
            hc_type (list): High contrast function
        '''
        
        # Check that inputs are of type 'list'
        self._islist(observation_files, target_res_obs, res_cont, hc_type)
        
        for indobs, obs in enumerate(observation_files):
            
            self._obs_file[indobs] = obs
            self._obs_data[indobs] = self.extract_observation(indobs)
            
            obs_name = self.obs_name[indobs]
            
            self._obs_data[indobs] = self.adapt_observation(self._obs_data[indobs], self.obs_name[indobs], target_res_obs[indobs], res_cont[indobs], wav_cont[indobs], hc_type[indobs])
            
            # Save the data
            self._logger.info(f'Save observation file {obs}')
            np.savez(os.path.join(global_params.paths.result_path, f'spectrum_obs_{self.obs_name[indobs]}.npz'), **self.obs_data[indobs])
            
            # if not self.adapted:   # If the model is not already adapted to the data, or if the user wants to redo the adaptation
            #     self._logger.info(f'adapt observation file {self.obs_file[indobs]}')
                
            #     # Setup target wavelength and resolution for the observation and the model
            #     if global_params.target_res_mod[indobs % len(global_params.target_res_mod)] == 'mod': # Kepping the model's resolution
            #         target_wavelength, target_resolution = self.model.wavelength, self.model.resolution
            #     if global_params.target_res_mod[indobs % len(global_params.target_res_mod)] == 'obs': # Using the observation's resolution except where its higher than the model's
            #         target_wavelength, target_resolution =    self.obs_data[indobs]['wav_spectro'], self.obs_data[indobs]['res_spectro']
            #     else:                                             # Using a custom resolution except where its higher than the model's
            #         target_wavelength, target_resolution = self.model.wavelength, np.full(len(self.model.wavelength), float(global_params.target_res_mod[indobs]))
   
            #     # # Masks to have larger cuts of the spectroscopic grid if needed (if rv is defined)
            #     if global_params.rv[indobs*3 % len(global_params.rv)] == 'NA':
            #         mask_mod_obs = (self.model.adapt_grid[indobs]['wavelength'] <= self.obs_data[indobs]['wav_spectro'][-1]) & (target_wavelength >= self.obs_data[indobs]['wav_spectro'][0]) 
            #         target_wavelength = target_wavelength[mask_mod_obs]
            #         target_resolution = target_resolution[mask_mod_obs]
            #     else:
            #         mask_mod_obs = (target_wavelength <= 1.01 * self.obs_data[indobs]['wav_spectro'][-1]) & (target_wavelength >= 0.99 * self.obs_data[indobs]['wav_spectro'][0])   # 1.01 corresponds to a value of 3000 km/s for the RV so we do no risk to lose data on the edges when applying the RV correction
            #         target_wavelength = target_wavelength[mask_mod_obs]
            #         target_resolution = target_resolution[mask_mod_obs]

            #     # Interpolate the resolution of the model onto the wavelength of the data to properly decrease the resolution if necessary
            #     interp_mod_to_obs = interp1d(self.model.wavelength, self.model.resolution, fill_value='extrapolate')
            #     target_resolution = interp_mod_to_obs(target_wavelength)
                
            #     shape = self.model.grid.values.shape
            #     pbar = tqdm(total=np.prod(shape), leave=False)

            #     def update(*a):
            #         pbar.update()

            #     try: # Parallel if possible
            #         ncpu = mp.cpu_count()
            #         with ThreadPool(processes=ncpu) as pool:
            #             for idx in np.ndindex(shape):
            #                 pool.apply_async(self.model.adapt_to_observation, args=(idx, self.obs_name[indobs], target_resolution, target_wavelength, remove_continuum), callback=update)

            #             pool.close()
            #             pool.join()
            #     except:
            #         for idx in np.ndindex(shape):
            #             self.model.adapt_to_observation(idx, self.obs_name[indobs], target_resolution, target_wavelength, remove_continuum)
            #             update()

                # print()
                # print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
                # print("-> To compare synthetic spectra with the observation we need to manage them. The following actions are performed:")
                # print("- extraction -")
                # print("- resizing on the observation's wavelength range -")
                # print("- adjustement of the spectral resolution -")
                # print("- substraction of the continuum (if needed) -")
                # print()
                # print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
                # print(f"-> Sarting the adaptation of {obs_name}")

                # adapt_grid(global_params, obs_dict, res_mod_nativ_interp, target_wav_mod, target_res_mod, obs_name, indobs)
            
            
        
   
# These lines are just for testing purposes. They will be removed for the final version
config = '/Users/allandenis/These/ForMoSA_Main/51_Eri/config_51Eri_b_ExoREM_all_spectro.ini'
model_path = '/Users/allandenis/test.nc'

analysis = Analysis(config, log_level='info')
global_params = GlobalParams(config)
analysis.adapt(global_params.paths.observation_files, target_res_obs=global_params.target_res_obs, res_cont=global_params.res_cont, wav_cont=global_params.wav_cont, hc_type=global_params.hc_type)