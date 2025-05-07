from ForMoSA.ModelGrid import ModelGrid
from ForMoSA.observation import Observation

class Nested_Sampling(object):
    '''
    ForMoSA Nested_Sampling class, which provided easy access to the parameters of the nested sampling algorithm
    
    Parameters
    ----------
    
    '''
    def __init__(self, grid: ModelGrid, observation: Observation, algorithm: str, logL_function: list):
        self._grid = grid
        self._observation = observation
        self._algorithm = algorithm
        self._logL_function = logL_function
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
        return f'Nested sampling, algorithm = {self.algorithm}'
    
    def __format__(self) -> str:
        return self.__repr__()    
    
   
    ##################################################
    # Properties
    ##################################################
    
    @property 
    def grid(self):
        return self._grid 
    
    @property 
    def observation(self):
        return self._observation
    
    @property 
    def algorithm(self):
        return self._algorithm
    
    @property 
    def logL_function(self):
        return self._logL_function
