import ForMoSA
from ForMoSA.ModelGrid import ModelGrid
from ForMoSA.observation import Observation


class ForMoSAError(Exception):
    pass


class Parameter(object):
    '''
    ForMoSA Parameter class. Handles a single parameter for the nested sampling algorithm
    '''

    VALID_PRIORS = {'uniform', 'gaussian', 'log-uniform', 'constant'}

    def __init__(self, name: str, prior: str, bounds: list = None, mean: float = None, std: float = None, value: float = None):
        if prior not in self.VALID_PRIORS:
            raise ForMoSAError(f" Prior '{prior}' not valid for the parameter '{name}'. "
                             f" Choose amongst : {', '.join(self.VALID_PRIORS)}.")
        
        if prior in {'uniform', 'log-uniform'}:
            if bounds is None:
                raise ForMoSAError(f" Prior '{prior}' needs bounds [min, max].")
            if not (isinstance(bounds, list) and len(bounds) == 2):
                raise ForMoSAError(" Bounds have to be a list [min, max] with uniform or log uniform priors.")

        if prior == 'gaussian':
            if mean is None or std is None:
                raise ForMoSAError(" Gaussian prior needs argument 'mean' and 'std'.")
                
        if prior == 'constant':
            if value is None:
                raise ForMoSAError(" Constant prior needs argument 'value'.")
        
        self._name = name
        self._prior = prior
        self._bounds = bounds
        self._mean = mean
        self._std = std
        self._value = value
        
        
    ##################################################
    # Representation
    ##################################################
        
    def __repr__(self):
        return f"Parameter(name={self.name}, prior={self.prior}, bounds={self.bounds}, " \
               f"mean={self.mean}, std={self.std}, value={self.value}"
               

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
    def is_fixed(self):      # Whether the parameter is fixed
        return self.prior == 'constant'


class NestedSampling_Params(object):
    '''
    ForMoSA NestedSampling_Params class. Handles dynamically the parameters of the nested sampling.
    
    Parameters
    ----------
    config_dict (dict): Configuration file dictionary
    
    Authors: Allan Denis
    '''
    
    def __init__(self, config_dict: dict):
        self._parameters = {}
        self.from_config(config_dict)
        

    def add_parameter(self, param: Parameter, name):
        '''
        Method to add a parameter used in the nested sampling algorithm

        Parameters
        ----------
        param : Instance of :class:`~ForMoSA.Parameter`
        name  : Name of the parameter

        Authors: Allan Denis
        '''
        
        if name in self._parameters:
            raise ForMoSAError(f" Parameter '{name}' already exists.")
        self._parameters[name] = Parameter(name, param.prior, param.bounds, param.mean, param.std, param.value)


    def from_config(self, config_dict: dict):
        '''
        Method to build the dictionnary of parameters from the config file dictionary

        Parameters
        ----------
        config_dict (dict): Configuration file dictionary

        Authors: Allan Denis
        '''
        for param_type in ['grid_parameters', 'physical_parameters']:    # Retrieve 'grid parameters' and 'physical parameters' objects
            for name, param in config_dict[param_type].items():        # (e.g. 'par1', 'rv_0", 'vsini_1')
                self.add_parameter(param, name)


    ##################################################
    # Representation
    ##################################################

    def __repr__(self):
        return f"NestedSampling_Params({self._parameters})"
    
    
    ##################################################
    # Properties
    ##################################################

    @property 
    def parameters(self) -> dict:            # Parameters
        return self._parameters

    @property 
    def list_params_names(self) -> list:     # List of parameters names
        return list(self.parameters)
    
    @property 
    def list_params_values(self) -> list:    # List of parameters values
        return list(self.parameters.values())
    
    @property 
    def n_free_parameters(self) -> int:      # Number of free parameters
        return sum(1 for p in self.parameters.values() if not p.is_fixed)
    
    @property 
    def n_fixed_parameters(self) -> int:     # Number of fixed parameters
        return len(self.list_params_names) - self.n_free_parameters
    
    @property 
    def free_parameters(self) -> dict:       # Dictionary of free parameters
        return {name: p for name, p in self._parameters.items() if not p.is_fixed}
    
    @property 
    def fixed_parameters(self) -> dict:      # Dictionary of fixed parameters
        return {name: p for name, p in self._parameters.items() if p.is_fixed}



class NestedSampling(object):
    '''
    ForMoSA Nested_Sampling class, which provides easy access to the parameters of the nested sampling algorithm
    
    Parameters
    ----------
    grid          (ModelGrid): Instance of :class:`~ForMoSA.ModelGrid`
    observation (Observation): Instance of :class:`~ForMoSA.Observation`
    
    Authors: Allan Denis 
    '''
    def __init__(self, grid: ModelGrid, observation: Observation, logger, params: NestedSampling_Params):
        self._grid = grid
        self._observation = observation
        self._logger = logger
        self._params = params
        self._results = None
        
    def run(self):
        """
        Run the nested sampling algorithm using the model, observation and nested sampling parameters.
        """
        
        

    def summary(self):
        return self.results.summary() if self.results else "No results yet."
    
   
    ##################################################
    # Representation
    ##################################################
    
    def __repr__(self):
        return f'Nested sampling'
    
    def __format__(self) -> str:
        return self.__repr__()    
    
   
    ##################################################
    # Properties
    ##################################################
    
    @property 
    def grid(self):            # Grid
        return self._grid 
    
    @property 
    def observation(self):     # Observation
        return self._observation
    
    @property 
    def algorithm(self):       # Algorithm
        return self._algorithm
    
    @property 
    def logL_function(self):   # logL function
        return self._logL_function
    
    @property 
    def n_obs(self):           # Number of observations
        return self.grid.n_obs
    
    @property 
    def params(self):          # Parameters
        return self._params
    

