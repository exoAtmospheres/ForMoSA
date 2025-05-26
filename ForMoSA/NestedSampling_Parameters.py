import numpy as np
import ForMoSA.utils_prior_functions as prior_functions


class ForMoSAError(Exception):
    pass

class Parameter(object):
    '''
    ForMoSA Parameter class. Handles a single parameter for the nested sampling algorithm
    
    Parameters
    ----------
    name              (str): Name of the parameter ('par1', 'par2', 'rv', 'd', ...)
    prior             (str): Prior function of the parameter ('uniform', 'gaussian', 'constant', 'log-uniform', 'computed')
    bounds           (list): Bounds of the prior (for the 'uniform' and 'log-uniform' priors)
    mean            (float): Mean of the prior (for the 'gaussian' prior)
    std             (float): Standard deviation of the prior (for the 'gaussian' prior)
    value           (float): Value of the prior (for the 'constant' prior)
    vsini_function    (str): vsini function used for the prior (only is name starts with vsini)
    
    Authors: Allan Denis
    '''

    def __init__(self, name: str, prior: str, bounds: list = None, mean: float = None, std: float = None, value: float = None, vsini_function: str = None):
        valid_priors = {'uniform', 'gaussian', 'log-uniform', 'constant', 'computed'}
        if prior not in valid_priors:
            msg = f" Prior '{prior}' not valid for the parameter '{name}'. Choose amongst : {', '.join(valid_priors)}."
            raise ForMoSAError(msg)
        
        if prior in {'uniform', 'log-uniform'}:
            if bounds is None:
                msg = f" Prior '{prior}' needs bounds [min, max]."
                raise ForMoSAError(msg)
            if not (isinstance(bounds, list) and len(bounds) == 2):
                msg = " Bounds have to be a list [min, max] with uniform or log uniform priors."
                raise ForMoSAError(msg)
            if (prior == 'log-uniform') and (bounds[0] <= 0 or bounds[1] <= 0):
                msg = " You cannot use negative bounds with log-uniform priors."
                raise ForMoSAError(msg)

        if prior == 'gaussian':
            if mean is None or std is None:
                msg = " Gaussian prior needs argument 'mean' and 'std'."
                raise ForMoSAError(msg)
            if std <= 0:
                msg = " You cannot use negative standard deviations with Gaussian priors."
                raise ForMoSAError(msg)
                
        if prior == 'constant':
            if value is None:
                msg = " Constant prior needs argument 'value'."
                raise ForMoSAError(msg)
                
        if name.startswith('vsini') and vsini_function == None:
            msg = " 'vsini' parameter needs a vsini function."
            raise ForMoSAError(msg)
                   
        self._name = name
        self._prior = prior
        self._bounds = bounds
        self._mean = mean
        self._std = std
        self._value = value
        self._vsini_function = vsini_function
        self._theta = None
        
        
    ##################################################
    # Representation
    ##################################################
        
    def __repr__(self):
        return f"Parameter(name={self.name}, prior={self.prior}, bounds={self.bounds}, " \
               f"mean={self.mean}, std={self.std}, value={self.value}, vsini_function={self.vsini_function} \n"
               

    ##################################################
    # Properties
    ##################################################
    
    @property 
    def name(self):          # Name
        return self._name 
    
    @property 
    def prior(self):         # Prior type
        return self._prior 
    
    @property 
    def bounds(self):        # Bounds (for Uniform and log-Uniform priors)
        return self._bounds 
    
    @property
    def mean(self):          # Mean (for Gaussian priors)
        return self._mean 
    
    @property 
    def std(self):           # Standard deviation (for Gaussian priors)
        return self._std 
    
    @property 
    def value(self):         # Value (for Constant prior)
        return self._value 
    
    @property 
    def vsini_function(self):
        return self._vsini_function
    
    @property 
    def is_fixed(self):      # Whether the parameter is fixed
        return self.prior == 'constant'
    
    @property 
    def theta(self):         # Current value randomly picked by the nested sampling
        return self._theta
    
    
    ##################################################
    # Methods
    ##################################################
    
    def _apply_prior(self, theta: float) -> float:
        '''
        Method to apply prior to a parameter given a random value theta picked uniformely between [0, 1]

        Parameters
        ----------
        theta (float): Parameter value randomly picked by the nested sampling
        
        Returns:
            - float: Transformed prior value

        Authors: Allan Denis
        '''
        
        if self.prior == 'uniform':
            self._theta = prior_functions.uniform_prior(self.bounds, theta)
            
        if self.prior == 'gaussian':
            self._theta = prior_functions.gaussian_prior(self.mean, self.std, theta)
            
        if self.prior == 'log-uniform':
            self._theta = prior_functions.loguniform_prior(self.bounds, theta)
            
        return self.theta
    

class NestedSampling_Params(object):
    '''
    ForMoSA NestedSampling_Params class. Handles dynamically the parameters of the nested sampling.
    
    Parameters
    ----------
    logger : logger
    
    Authors: Allan Denis
    '''
    
    def __init__(self, logger):
        self._parameters = {}
        self._logger = logger
        

    ##################################################
    # Representation
    ##################################################

    def __repr__(self):
        return f'<NestedSampling_Params(\n{self._parameters})>'
    
    
    
    ##################################################
    # Properties
    ##################################################
    
    @property
    def parameters(self) -> dict:
        return self._get_parameters('all')
    
    @property
    def free_parameters(self) -> dict:
        return self._get_parameters('free')
    
    @property
    def fixed_parameters(self) -> dict:
        return self._get_parameters('fixed')
    
    @property 
    def list_params_keys(self) -> list:
        return list(self.parameters)
    
    @property 
    def list_free_params_keys(self) -> list:
        return list(self.free_parameters)
        
    @property 
    def list_params_names(self) -> list:
        return [p.name for p in self.parameters.values()]
    
    @property
    def list_free_params_names(self) -> list:
        return [p.name for p in self.free_parameters.values()]
    
    @property 
    def list_fixed_params_names(self) -> list:
        return [p.name for p in self.fixed_parameters.values()]
    
    @property
    def n_parameters(self) -> int:
        return len(self.parameters)
    
    @property
    def n_free_parameters(self) -> int:
        return len(self.free_parameters)
    
    @property
    def n_fixed_parameters(self) -> int:
        return len(self.fixed_parameters)
    
    @property
    def free_param_priors(self):
        return self._get_param_attr('prior', 'free')
    
    @property
    def free_param_bounds(self):
        return self._get_param_attr('bounds', 'free')
    
    @property 
    def theta(self):
        return self._theta

    
    
    ##################################################
    # Methods
    ##################################################
    
    
    def _add_parameter(self, param: Parameter, name) -> None:
        '''
        Method to add a parameter used in the nested sampling algorithm

        Parameters
        ----------
        param : Instance of :class:`~ForMoSA.Parameter`
        name  : Name of the parameter

        Authors: Allan Denis
        '''
        
        if name in self._parameters:
            msg = f" Parameter '{name}' already exists."
            self._logger.error(msg)
            raise ForMoSAError(msg)
    
        self._parameters[name] = Parameter(name, param.prior, param.bounds, param.mean, param.std, param.value, param.vsini_function)
    
    
    def _check_params(self) -> None:
        '''
        Method to check that there is no issue in the nested sampling parameters
        
        Authors: Allan Denis
        '''
         
        nb_vsini, nb_ld, nb_r, nb_d, nb_av, nb_bb_t, nb_bb_r = 0, 0, 0, 0, 0, 0, 0
         
        for key in self.parameters.keys():
            if key.startswith('vsini'):
                nb_vsini += 1
            if key.startswith('ld'):
                nb_ld += 1
            if key.startswith('r') and not(key.startswith('rv')):
                nb_r += 1
            if key.startswith('d'):
                nb_d += 1
            if key.startswith('av'):
                nb_av += 1
            if key.startswith('bb_R'):
                nb_bb_r += 1
            if key.startswith('bb_T'):
                nb_bb_t += 1
         
        if (nb_vsini != nb_ld):
            msg = " You need to define both vsini and limb darkening priors of set them both to 'NA'."
            self._logger.error(msg)
            raise ForMoSAError(msg)
        if (nb_r != nb_d):
            msg = " You need to define both radius and distance priors or set them both to 'NA'."
            self._logger.error(msg)
            raise ForMoSAError(msg)
        if nb_vsini > 1:
            self._logger.warning(' Multiples vsini priors are defined for different observations, which is very unlikely.')
        if nb_r > 1:
            msg = ' Multiples radius and distance priors are defined for different observations. Please use at most 1 prior for these parameters.'
            self._logger.error(msg)
            raise ForMoSAError(msg)
        if nb_av > 1:
            msg = ' Multiples interstellar extinction priors are defined for different observations. Please use at most 1 prior for this parameter.'
            self._logger.error(msg)
            raise ForMoSAError(msg)
        if nb_bb_t != nb_bb_r:
            msg = " You need to define both black body temperature and black body radius or set them both to 'NA'."
            self._logger.error(msg)
            raise ForMoSAError(msg)
        if nb_bb_t != nb_d and nb_bb_t != 0:
            msg = " If tou define a blackbody, you also need to define a distance."
            self._logger.error(msg)
            raise ForMoSAError(msg)
            
            
    def _get_param_value(self, name: str, theta: np.ndarray) -> float:
        '''
        Method to get the value of a parameter given its name.
    
        Parameters
        ----------
        name         (str): Name of the parameter (either a key like 'par1', 'rv', 'vsini_0' or a physical name like 'Teff', 'logg')
        theta (np.ndarray): List of parameter values corresponding to list_params_keys
    
        Returns:
            float: Current value of the parameter, either from theta or the constant prior value
    
        Authors: Allan Denis
        '''
        
       
        if not isinstance(theta, (list, np.ndarray)) or len(theta) == 0:
            raise ForMoSAError(f"theta must be a non-empty array, got: {theta}")
    
        theta = np.asarray(theta)
    
        # Determine parameter key
        if name in self.list_params_keys:
            param_key = name
            theta_index = self.list_free_params_keys
        elif name in self.list_params_names:
            param_key = self.list_params_keys[self.list_params_names.index(name)]
            theta_index = self.list_free_params_names
        else:
            msg = f"Invalid parameter name: {name}. Choose from {self.list_params_keys} or {self.list_params_names}"
            self._logger.error(msg)
            raise ForMoSAError(msg)
    
        # Get parameter
        p = self.parameters[param_key]
   
        # Handle constant prior
        if p.prior == 'constant':
            if not isinstance(p.value, (int, float)) or p.value is None:
                msg = f"Parameter {name} has invalid constant value: {p.value}"
                self._logger.error(msg)
                raise ForMoSAError(msg)
            value = float(p.value)
        else:
            try:
                idx = theta_index.index(name)
                value = theta[idx]
            except ValueError:
                msg = f"Parameter {name} not found in theta_index {theta_index}"
                self._logger.error(msg)
                raise ForMoSAError(msg)
    
        # Update parameter state
        p._theta = value
        return value
 
            
    def _get_parameters(self, param_type: str = 'all') -> dict:
        '''
        Returns all the parameters according to their type : 'all', 'free', 'fixed'.
    
        Parameters
        ----------
        param_type (str): Type of parameter to return ('all', 'fixed', 'free')
    
        Returns:
            - dict: Dictionary of parameters filtered.
        
        Authors: Allan Denis
        '''
        
        if param_type not in ['all', 'free', 'fixed']:
            raise ForMoSAError("param_type must be 'all', 'free', or 'fixed'")
    
        if param_type == 'free':
            return {k: p for k, p in self._parameters.items() if not p.is_fixed}
        elif param_type == 'fixed':
            return {k: p for k, p in self._parameters.items() if p.is_fixed}
        
        return self._parameters
    
    
    def _get_param_attr(self, attr: str, param_type: str = 'all') -> dict:
        '''
        Extract a given attribute for each parameter of a given type
    
        Parameters
        ----------
        attr        (str): Name of the attribute to extract
        param_type  (str): Type of parameter ('all', 'fixed', 'free')
    
        Returns:
            - dict: {param_name: attribute_value}
        
        Authors: Allan Denis
        '''
        
        return {k: getattr(p, attr) for k, p in self._get_parameters(param_type).items()}
