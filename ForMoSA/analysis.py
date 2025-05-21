import numpy as np 
import logging
import os
from scipy.interpolate import interp1d
import ForMoSA.utils as utils
import colorlog

from ForMoSA.global_params import GlobalParams
from ForMoSA.ModelGrid import ModelGrid
from ForMoSA.ForMoSAPaths import ForMoSAPaths
from ForMoSA.observation import Observation
from ForMoSA.nested_sampling import NestedSampling

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
        
        logger.propagate = False
        
        while logger.hasHandlers():
            logger.removeHandler(logger.handlers[0])
        
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


        # Inits
        global_params = GlobalParams(config_file_path)
        analysis._paths = paths
        analysis._observation = Observation(paths.observation_path, logger)
        analysis._grid = ModelGrid(paths.model_path, logger)
        analysis._adapted = adapted  
        analysis._nested_sampling = NestedSampling(global_params.config_inversion['ns_algo'], global_params.config_inversion['npoints'], logger, global_params.config_ns_algo)
        analysis._logger = logger
        
        # Build and check list of nested sampling parameters
        analysis._add_nested_sampling_parameters_from_config(global_params.config_parameters)

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
    def paths(self) -> str | os.PathLike:
        return self._paths
    
    @property 
    def grid(self) -> ModelGrid:
        return self._grid
    
    @property 
    def observation(self) -> Observation:
        return self._observation
    
    @property 
    def adapted(self) -> bool:
        return self._adapted
    
    @property  
    def nested_sampling(self) -> NestedSampling:
        return self._nested_sampling
    
    @property 
    def global_params(self) -> GlobalParams:
        return self._global_params
    
        
    ##################################################
    # Methods
    ##################################################
    
    
    def adapt(self, observation_files: list, res_obs: list=['obs'], res_mod: list=['obs'], res_cont: list=['NA'], wav_cont: list=['NA'], hc_type: list=['NA'], emulator: str='NA', interp_method: list=['linear']):
        '''
        Method to adapt the grid of model to each observation

        Parameters
        ----------
            observation_files (list): List of the observation files
            res_obs           (list): Target resolution of the observation ('obs', 'mod' or float)
            res_mod           (list): Target resolution of the model ('obs' or float)
            res_cont          (list): Resolution of the continuum
            wav_cont          (list): Wavelength range on which we want to estimate the continuum
            hc_type           (list): High contrast function
            emulator          (list): Method of adaptation of the grid ('NA', 'PCA' of 'NMF')
            
        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''
        
        # Check that inputs are of type 'list'
        is_not_list = utils.check_format(observation_files, res_obs, res_mod, res_cont, hc_type, type_expected=list)
        if len(is_not_list) > 0:
                msg = f" Params in wrong format : {', '.join(is_not_list)}."
                self._logger.critical(msg)
                raise ForMoSAError(msg)
        
        for indobs, obs in enumerate(observation_files):
            
            obs_data_spectro = self.observation.obs_data[indobs]['spectro']
            obs_data_photo = self.observation.obs_data[indobs]['photo']
            obs_name = self.observation.obs_name[indobs]
            
            self._logger.info(f' Current observation: {obs_name}.')
   
            target_res_obs = self._set_obs_target_resolution(res_obs[indobs % len(res_obs)], obs_data_spectro['wav'], obs_data_spectro['res'])
            
            # Determine continuum types
            star_continuum, remove_continuum = self._determine_continuum_types(hc_type[indobs % len(hc_type)], res_cont[indobs % len(res_cont)])

            if len(target_res_obs) > 0:
                self.observation._adapt_observation(target_res_obs, res_cont[indobs % len(res_cont)], wav_cont[indobs % len(wav_cont)], star_continuum, indobs)
                self._logger.info(f' Observation {obs_name} adapted.')
            
            if not self.adapted:   # If the model is not already adapted to the data, or if the user wants to redo the adaptation
                ins_spectro, wavelength_photo, ins_photo = obs_data_spectro['ins'], obs_data_photo['wav'], obs_data_photo['ins']
                
                # Determine target wavelength and resolution of the model
                target_wavelength, target_resolution = self._determine_grid_target_wavelength_and_resolution(obs_data_spectro['wav'], obs_data_spectro['res'], res_mod[indobs % len(res_mod)])
                # Adapt grid using target wavelength and resolution
                self.grid.adapt_grid(target_resolution, target_wavelength, wavelength_photo, ins_spectro, ins_photo, wav_cont[indobs % len(wav_cont)], res_cont[indobs % len(res_cont)], remove_continuum, obs_name)

                if emulator == 'PCA':
                    self._logger.info(' Decomposing the grid using PCA')
                    
                    # TODO
                    
                if emulator == 'NMF':
                    self._logger.info(' Decomposing the grid using NMF')
                    
                    # TODO
                    
        
        # Save the data
        self.observation._save_all_observations(self.paths.result_path)
        
        self.grid._interpolate_missing_values()
        self.grid._save_grid(self.paths.adapt_store_path)
        

    def _set_obs_target_resolution(self, target_res_obs : str, wav_obs_spectro: np.ndarray, res_obs_spectro: np.ndarray) -> np.ndarray:
        '''
        Method to set the target resolution of the observation

        Parameters
        ----------
        target_res_obs (str | float): Target resolution of the observation ('obs' or float) 
        res_mod_obs     (np.ndarray): Wavelength grid of the observation
        res_obs_spectro (np.ndarray): Resolution of the observation

        Returns:
            target_res_obs (np.ndarray): Target resolution of the observation

        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''
        
        # Setup target resolution for the observation
        # Interpolate the resolution of the model onto the wavelength of the data to properly decrease the resolution if necessary
        interp_mod_to_obs = interp1d(self.grid.wavelength, self.grid.resolution, fill_value='extrapolate')
        res_mod_obs = interp_mod_to_obs(wav_obs_spectro)

        if target_res_obs == 'obs': # Keeping the resolution of the observation except where its higher than the model's
            target_res_obs = np.minimum(res_obs_spectro, res_mod_obs)
        else:                                             # Using a custom resolution except where its higher than the model's or the observation's
            res_custom = np.full(len(res_obs_spectro), float(target_res_obs))
            target_res_obs = np.minimum(res_obs_spectro, res_mod_obs, res_custom)
            
        return target_res_obs


    def _determine_grid_target_wavelength_and_resolution(self, wav_obs_spectro: np.ndarray, res_obs_spectro: np.ndarray, target_res_mod: np.ndarray, indobs: int=0) -> tuple[np.ndarray, np.ndarray]:
        '''
        Method to set target wavelength and resolutions of the model.
        This depends on the wavelength and resolution of the current observation we want to adapt the model to.

        Parameters
        ----------
        wav_obs_spectro  (np.ndarray): Wavelength grid of the observation
        res_obs_spectro  (np.ndarray): Resolution of the observation
        target_res_mod  (str | float): Target resolution ('obs', 'mod' or float)
        indobs                  (int): Index of current observation

        Returns:
            - target_wavelength (np.ndarray): Target wavelength of the model 
            - target_resolution (np.ndaraay): Target resolution of the model
            
        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''
        
        # Setup target wavelength and resolution for the observation and the model
        if target_res_mod == 'mod': # Kepping the model's resolution
            target_wavelength, target_resolution = self.grid.wavelength, self.grid.resolution
        if target_res_mod == 'obs': # Using the observation's resolution except where its higher than the model's
            target_wavelength, target_resolution = wav_obs_spectro, res_obs_spectro
        else:                                             # Using a custom resolution except where its higher than the model's
            target_wavelength, target_resolution = self.grid.wavelength, np.full(len(self.grid.wavelength), float(target_res_mod))

        if len(wav_obs_spectro) > 0:
            # # Masks to have larger cuts of the spectroscopic grid if needed (if rv is defined)
            if ('rv' in self.nested_sampling.params.parameters.keys()) or (f'rv_{indobs}' in self.nested_sampling.params.parameters.keys()):
                mask_mod_obs = (target_wavelength <= 1.01 * wav_obs_spectro[-1]) & (target_wavelength >= 0.99 * wav_obs_spectro[0])   # 1.01 corresponds to a value of 3000 km/s for the RV so we do no risk to lose data on the edges when applying the RV correction
                target_wavelength, target_resolution = target_wavelength[mask_mod_obs], target_resolution[mask_mod_obs]
            else:
                mask_mod_obs = (target_wavelength <= wav_obs_spectro[-1]) & (target_wavelength >= wav_obs_spectro[0]) 
                target_wavelength, target_resolution = target_wavelength[mask_mod_obs], target_resolution[mask_mod_obs]
        
        return target_wavelength, target_resolution
                
    
    @staticmethod
    def _determine_continuum_types(hc_type: str, res_cont: str | float) -> str:
        '''
        Method to determine the star continuum type ("estimate", "remove", "NA") from the high contrast function type

        Parameters
        ----------
        hc_type          (str): high-contrast function type
        res_cont (str | float): Resolution of the continuum

        Returns:
            - star_continuum    (str): star_continuum type ("estimate", "remove" or "NA")
            - remove_continuum (bool): Whether to remove the continuum of the grid models

        Authors: Allan Denis
        '''
        
        if res_cont != 'NA':
            if hc_type != 'NA':
                star_continuum = 'estimate'
                remove_continuum = False
            else:
                star_continuum = 'remove'
                remove_continuum = True
        else:
            star_continuum = 'NA'
            remove_continuum = False
        
        return star_continuum, remove_continuum
        
        
    def launch_nested_sampling(self, logL_function: list, wav_for_fitting: list, wav_cont: list = ['NA'], res_cont: list=['NA'], hc_type: list=['NA'], bounds_lsq: list = ['NA'], interp_method: str = 'linear', emulator: list = ['NA']):
        '''
        Method to launch the nested sampling algorithm

        Parameters
        ----------
        algorithm        (str): Algorithm to be used ('nestle', 'ultranest', 'pymultinest')
        logL_function   (list): logL type for each observation
        wav_for_fitting (list): Wavelength grid used for fitting for each observation
        npoints          (int): Number of living points used in the nested sampling algorithm
        wav_cont        (list): Wavelength grid used for the continuum for each observation
        res_cont        (list): Resolution used for the continuum for each observation
        bounds_lsq      (list): Least Squares bounds used for each observation (for high-contrast observations only)
        interp_method    (str): Interpolation method to interpolate between the gridpoints

        Authors: Allan Denis
        '''
        
        # Check that inputs are of type 'list'
        is_not_list = utils.check_format(logL_function, wav_for_fitting, wav_cont, res_cont, hc_type, bounds_lsq, type_expected=list)
        if len(is_not_list) > 0:
                msg = f" Params in wrong format : {', '.join(is_not_list)}."
                self._logger.critical(msg)
                raise ForMoSAError(msg)
                
        # Load adapted observations and grids
        self.observation._load_adapted_observations_from_files(self.paths.result_path)
        self.grid._load_grid_from_files(self.paths.adapt_store_path, self.observation.obs_name_list)
        
        # Run nested sampling
        self.nested_sampling.run(logL_function, self.paths.result_path, self.observation, self.grid, interp_method=interp_method, wav_cont=wav_cont, res_cont=res_cont, bounds_lsq=bounds_lsq, emulator=emulator)

        # Savings
        self.nested_sampling._save_results(self.paths.result_path)
        

    def _add_nested_sampling_parameters_from_config(self, config_dict: dict) -> None:
        '''
        Method to create the list of nested sampling parameters from the configuration file dictionary
        
        Parameters
        ----------
        config_dict (dict): Dictionary of the nested sampling config parameters
        
        Authors: Allan Denis
        '''
        
        for param_type in ['grid_parameters', 'physical_parameters']:    # Retrieve 'grid parameters' and 'physical parameters' objects
            for name, param in config_dict[param_type].items():        # (e.g. 'par1', 'rv_0", 'vsini_1')
                self.nested_sampling.params._add_parameter(param, name)
           
        # Additional step to check for the global consistance of the parameters
        self.nested_sampling.params._check_params()
        
        # Replace 'parX' names by associated physical parameters ('Teff', 'logg', ...)
        for name in self.nested_sampling.params.parameters.keys():
            if name.startswith('par'):  # Detect grid parameters
                self.nested_sampling.params.parameters[name]._name = self.grid.titles[self.grid.keys.index(name)]  # Rename parameter with title associated to 'parX'
                