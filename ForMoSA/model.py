import numpy as np 
import logging
from pathlib import Path
import os 
import xarray as xr
from ForMoSA.utils_spec import resolution_decreasing, continuum_estimate


class Model(object):
    '''
    Model class, which provides easy access to a grid of model
    
    Parameters
    ----------
    path : str | os.PathLike
        Path to the grid of model.
    '''
    
    def __init__(self, path: str | os.PathLike) -> None:
        
        # the command .expanduser() transforms the path in a full absolute path, removing any '~' in the path
        self._root = Path(path).expanduser().parent
        self._name = str(Path(path).expanduser()).split('/')[-1].split('.nc')[0]
        self._adapted_grid = dict()
        self.load_model()
        
    ##################################################
    # Representation
    ##################################################

    def __repr__(self) -> str:
        return f'<Model, name={self.name}, root={self.root}, wavelength={min(self.wavelength):.3f}-{max(self.wavelength):.3f}, resolution={min(self.resolution):.3f}-{max(self.resolution):.3f}>'

    def __format__(self) -> str:
        return self.__repr__()

    ##################################################
    # Properties
    ##################################################

    @property  
    def root(self): 
        return self._root

    @property  
    def name(self):
        return self._name 
    
    @property 
    def path(self):
        return Path(str(self.root / self.name) + '.nc')
    
    @property  
    def attrs(self):
        return self._attrs
    
    @property  
    def wavelength(self):
        return self._wavelength
    
    @property  
    def nyquist(self):      # Nyquist resolution
        return self.wavelength / (2 * np.diff(self.wavelength, append=(2*self.wavelength[-1] - self.wavelength[-2])))
    
    @property  
    def resolution(self):   # Resolution defined as the minimum between the given resolution and the Nyquist resolution
        return np.minimum(self.nyquist, self._resolution)
    
    @property  
    def grid(self):
        return self._grid
    
    @property  
    def adapted_grid(self):
        return self._adapted_grid

    
    ##################################################
    # Methods
    ##################################################
    
    def load_model(self):
        '''
        Read the model grid and store important information
        '''
        
        ds = xr.open_dataset(self.path, decode_cf=False, engine="netcdf4")
        self._wavelength = ds['wavelength'].values
        self._resolution = ds.attrs['res']
        self._attrs = ds.attrs
        self._attrs['res'] = self.resolution
        self._grid = ds['grid']
        
        
    def add_grid(self, indobs: int, obs_name: str, wavelength: np.ndarray, resolution: np.ndarray, grid: np.float64):
        '''
        Add a grid correspondong to< the observation obs_name

        Parameters
        ----------
        indobs : int
            Observation number
        obs_name : str
            Name of the observation
        wavelength : np.ndarray
            wavelength grid of the model
        resolution : np.ndarray
            resolution grid of the model
        grid : np.float64
            grid of the model
        '''
        self._adapted_grid[indobs] = {'obs_name': obs_name, 'wavelength': wavelength, 'resolution': resolution, 'grid': grid}
        
        
    def adapt_to_observation(self, idx, obs_name: str, target_resolution: np.float64, target_wavelength: np.float64, remove_continuum: bool=False):
        '''
        Adapt the grid of models to a given resolution.
    
        Parameters
        ----------
        idx : (tuple)
            Index of the current model
        obs_name : str
            Name of the observation
        target_resolution : np.float64
            Target resolution to reach
        target_wavelength : np.float64
            Target wavelength to reach
        remove_continuum : bool
            Whether to remove the continuum
        '''
        
        
        
        self._adapted_grid[obs_name] = {
            'wavelength': self._wavelength,
            'flux': degraded_flux,
            'resolution': target_resolution
        }
        
    
    def degrade_resolution(self, grid_values: np.ndarray, target_resolution: np.float64, target_wavelength: np.float64):
        '''
        Degrade the resolution of a grid to a given resolution        

        Parameters
        ----------
        grid_values : np.ndarray
            grid values
        target_resolution : np.float64
            Target resolution to reach
        target_wavelength : np.float64
            Target wavelength to reach
        '''
        
        
    
        return

        
    
        
        
        
# These lines are just for testing purposes, they will be removed for the final version
model = Model('/Users/allandenis/These/ForMoSA_Main/INPUT_MODELS/EXOREM_native.nc')
