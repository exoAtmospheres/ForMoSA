import logging
import ForMoSA.utils as utils
import colorlog

from ForMoSA.global_params import GlobalParams
from ForMoSA.ModelGrid import ModelGrid
from ForMoSA.ForMoSAPaths import ForMoSAPaths
from ForMoSA.Observation import Observation
from ForMoSA.NestedSampling import NestedSampling
from ForMoSA.NestedSampling_Plotting import NestedSampling_Plotting

# log
_log = logging.getLogger(__name__)

class ForMoSAError(Exception):
    pass

class Analysis(object):
    '''
    ForMoSA data analysis class
    
    Parameters
    ----------
    global_params                 (dict): Dictionary of global parameters
    adapted                       (bool): Whether the model is adapted to the data, by default False. Can be set to True if the model has already been adapted to the data
    fitted                        (bool): Whether the data have already been fitted for
    log_level (str): og level of the handler, by default ``'info'`` for all important informations.
    
    Authors: Allan Denis
    '''
    
    def __init__(self, global_params: GlobalParams, adapted: bool = False, fitted: bool = False, log_level: str = 'info') -> 'Analysis | None' :

        logger = logging.getLogger("ForMoSA")
        
        logger.propagate = False
        
        while logger.hasHandlers():
            logger.removeHandler(logger.handlers[0])
        
        logger.setLevel(log_level.upper())
        
        # File handler (no color)
        file_handler = logging.FileHandler(global_params.paths._result_path / 'analysis.log', mode='w', encoding='utf-8')
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
        self._config_params = global_params.config_params
        self._paths = global_params.paths
        self._adapted = adapted  
        self._fitted = fitted
        self._ns = NestedSampling(self.config_params['inversion']['ns_algo'], self.config_params['inversion']['npoints'], logger, self.config_params['ns_algo'])
        print(self.config_params['plottings'])
        self._ns._plotting = NestedSampling_Plotting(logger, self.config_params['plottings'])
        self._logger = logger
        
        # Build and check list of nested sampling parameters
        self._add_NestedSampling_parameters_from_config(self.config_params['parameters'])
    
    ##################################################
    # Representation
    ##################################################
    
    def __repr__(self) -> str:
        return f'<Analysis, config_file={self.paths.config_file_path}>'

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
    def adapted(self) -> bool:
        return self._adapted
    
    @property  
    def ns(self) -> NestedSampling:
        return self._ns
    
    @property 
    def config_params(self) -> dict():
        return self._config_params
    
    @property 
    def paths(self) -> ForMoSAPaths:
        return self._paths
    
    @property 
    def observation(self) -> Observation:
        return self.paths.observation
    
    @property
    def grid(self) -> ModelGrid:
        return self.paths.grid
    
    @property 
    def fitted(self) -> bool:
        return self._fitted
    
        
    ##################################################
    # Methods
    ##################################################
    
    
    def adapt(self):
        '''
        Method to adapt the grid of model to each observation
         
        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''
        
        adapt = self.config_params['adapt']
        inversion = self.config_params['inversion']
        
        res_obs   = adapt['target_res_obs']
        res_mod   = adapt['target_res_mod']
        res_cont  = adapt['res_cont']
        wav_cont  = adapt['wav_cont']
        emulator  = adapt['emulator']
        hc_type   = inversion['hc_type']
        
        # Parameters we want to check the format
        params = {
            'res_obs': res_obs,
            'res_mod': res_mod,
            'res_cont': res_cont,
            'wav_cont': wav_cont,
            'emulator': emulator,
            'hc_type': hc_type,
        }
        
        n_obs = self.observation.n_obs
        
        wrong_type = [name for name, val in params.items() if not isinstance(val, list)]
        wrong_length = [name for name, val in params.items() if not (len(val) == 1 or len(val) == n_obs)]
        
        # Errors
        if wrong_type:
            msg = f"Params not list: {', '.join(wrong_type)}."
            self._logger.critical(msg)
            raise ForMoSAError(msg)
        
        if wrong_length:
            msg = f"Params with wrong length (must be 1 or {n_obs}): {', '.join(wrong_length)}."
            self._logger.critical(msg)
            raise ForMoSAError(msg)

        self.observation.adapt_all_observations(res_obs, self.grid.wavelength, self.grid.resolution, res_cont = res_cont, wav_cont = wav_cont, hc_type = hc_type)
    
        if not self.adapted:   # If the model is not already adapted to the data, or if the user wants to redo the adaptation
            # Adapt grid using target wavelength and resolution
            self.grid.adapt_all_grids(self.observation.obs_data, res_mod, self.ns.params, wav_cont = wav_cont, res_cont = res_cont, hc_type = hc_type)
    
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
        else:
            # Load adapted observations and grids
            self.observation._load_adapted_observations_from_files(self.paths.result_path)
            self.grid._load_grid_from_files(self.paths.adapt_store_path, self.observation.obs_name_list)
            
        # grid and data are now adapted
        self._adapted = True
                 
        
    def nested_sampling(self):
        '''
        Method to launch the nested sampling algorithm

        Parameters
        ----------
        algorithm        (str): Algorithm to be used ('nestle', 'ultranest', 'pymultinest')
        logL_function   (list): logL type for each observation
        wav_for_fitting (list): Wavelength grid used for fitting for each observation
        wav_cont        (list): Wavelength grid used for the continuum for each observation
        res_cont        (list): Resolution used for the continuum for each observation
        bounds_lsq      (list): Least Squares bounds used for each observation (for high-contrast observations only)
        interp_method    (str): Interpolation method to interpolate between the gridpoints

        Authors: Allan Denis
        '''
        
        adapt = self.config_params['adapt']
        inversion = self.config_params['inversion']
        
        res_obs       = adapt['target_res_obs']
        res_mod       = adapt['target_res_mod']
        res_cont      = adapt['res_cont']
        wav_cont      = adapt['wav_cont']
        emulator      = adapt['emulator']
        interp_method = adapt['method']
        
        hc_type        = inversion['hc_type']
        logL_function  = inversion['logL_type']
        wav_for_fitting = inversion['wav_fit']
        bounds_lsq     = inversion['hc_bounds_lsq']

        # Check that inputs are of type 'list'
        is_not_list = utils.check_format(res_obs, res_mod, res_cont, wav_cont, emulator, hc_type, logL_function, wav_for_fitting, bounds_lsq, type_expected=list)
        if len(is_not_list) > 0:
                msg = f" Params in wrong format : {', '.join(is_not_list)}."
                self._logger.critical(msg)
                raise ForMoSAError(msg)
                
        if not(self.fitted):
            # Run nested sampling
            self.ns.run(logL_function, self.paths.result_path, self.paths.observation, self.paths.grid, interp_method=interp_method, wav_cont=wav_cont, res_cont=res_cont, bounds_lsq=bounds_lsq, emulator=emulator)
    
            # Savings
            self.ns._save_results(self.paths.result_path)
        
        else:
            self.ns._load_results(self.paths.result_path)
            
        # Summary
        print(self.ns._summary())
        
        self.ns._compute_best_model(self.paths.observation, self.paths.grid, interp_method = interp_method, wav_cont = wav_cont, res_cont = res_cont, bounds_lsq = bounds_lsq, hc_type = hc_type)
        

    def _add_NestedSampling_parameters_from_config(self, config_dict: dict) -> None:
        '''
        Method to create the list of nested sampling parameters from the configuration file dictionary
        
        Parameters
        ----------
        config_dict (dict): Dictionary of the nested sampling config parameters
        
        Authors: Allan Denis
        '''
        
        for param_type in ['grid_parameters', 'physical_parameters']:    # Retrieve 'grid parameters' and 'physical parameters' objects
            for name, param in config_dict[param_type].items():        # (e.g. 'par1', 'rv_0", 'vsini_1')
                self.ns.params._add_parameter(param, name)
           
        # Additional step to check for the global consistance of the parameters
        self.ns.params._check_params()
        # Replace 'parX' names by associated physical parameters ('Teff', 'logg', ...)
        for name in self.ns.params.parameters.keys():
            if name.startswith('par'):  # Detect grid parameters
                self.ns.params.parameters[name]._name = self.grid.titles[self.paths.grid.keys.index(name)]  # Rename parameter with title associated to 'parX'
                
                
    def _plot(self, label_ins: str='no', trans: str='yes', uncert: str='yes') -> None:
        '''
        Method to use all the plotting methods

        Parameters
        ----------
        label_ins (str): Whether to label instruments in best fit plot
        trans     (str): Whether to plot the transmission filters
        uncert    (str): Whether to plot the uncertainties
        
        Authors: Allan Denis
        '''

        results = self.ns.results
        param_names = self.ns.params.list_free_params_names
        param_best_values = self.ns.param_best_dict
        modif_data = self.ns.modif_data
        best_model = self.ns.best_model
        
        self.ns.plotting._plot_corner(results, param_names)
        self.ns.plotting._plot_chains(results, param_names, param_best_values)
        self.ns.plotting._plot_radar(results, param_names)
        self.ns.plotting._plot_fit(modif_data, best_model, label_ins=label_ins, trans=trans, uncert=uncert)
                