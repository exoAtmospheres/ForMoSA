import numpy as np 
from pathlib import Path
import logging
import os 
import xarray as xr
from ForMoSA.utils_spec import resolution_decreasing, continuum_estimate
from scipy.interpolate import interp1d
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import multiprocessing as mp
from functools import partial


class ForMoSAError(Exception):
    pass

class Model(object):
    '''
    Model class, which provides easy access to a grid of model
    
    Parameters
    ----------
    path (str | os.PathLike): Path to the grid of model.
    log_level (str): Log level of the handler, by default ``'info'`` for all important informations.
    '''
    
    def __init__(self, path: str | os.PathLike, logger) -> None:
        
        # the command .expanduser() transforms the path in a full absolute path, removing any '~' in the path
        self._root = Path(path).expanduser().parent
        self._name = str(Path(path).expanduser()).split('/')[-1].split('.nc')[0]
        self._adapted_grid = dict()
        self.load_model()
        self._logger = logger
        
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
    
    @property 
    def counter(self):
        return len(self.adapted_grid) - 1

    
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
        
        
    def _add_adapted_grid(self, wavelength: np.ndarray, resolution: np.ndarray, grid: np.float64):
        '''
        Add a grid to be adapted to a specific wavelength and a specific resolution

        Parameters
        ----------
        indobs (int): Observation number
        wavelength (np.ndarray): wavelength grid of the model
        resolution (np.ndarray): resolution grid of the model
        grid (np.float64): grid of the model
        '''
        self._adapted_grid[self.counter+1] = {'wavelength': wavelength, 'resolution': resolution, 'grid': grid}
        
        
    def adapt_grid(self, target_resolution: np.ndarray, target_wavelength: np.ndarray, wav_cont: np.ndarray=[], res_cont: np.ndarray=[], remove_continuum: bool=False):
        '''
        Adapt the grid of models to a given resolution and wavelength.
    
        Parameters
        ----------
        target_resolution (np.float64): Target resolution to reach
        target_wavelength (np.float64): Target wavelength to reach
        remove_continuum (bool): Whether to remove the continuum
        '''
        
        grid = np.empty((len(target_wavelength),) + np.shape(self.grid.values)[1:])
        self._add_adapted_grid(target_wavelength, target_resolution, grid)
        
        interp_mod_to_obs = interp1d(self.wavelength, self.resolution, fill_value='extrapolate')
        resolution_model = interp_mod_to_obs(target_wavelength)
             
        shape = self.grid.values.shape[1:]
        pbar = tqdm(total=np.prod(shape), leave=False)

        def update_result(result, idx, model):
            model._adapted_grid[model.counter]['grid'][(..., ) + idx] = result
            pbar.update(1)
            
        try: # Parallel if possible
            ncpu = mp.cpu_count()
            with ThreadPool(processes=ncpu) as pool:
                for idx in np.ndindex(shape):
                    # Dans votre boucle, pour chaque idx :
                    callback = partial(update_result, idx=idx, model=self)
                    pool.apply_async(self.adapt_model, args=(self.grid.values[(..., ) + idx], target_resolution, target_wavelength, resolution_model, wav_cont, res_cont, remove_continuum), callback=callback)
                pool.close()
                pool.join()
                
        except:
            for idx in np.ndindex(shape):
                self._adapted_grid[self.counter]['grid'][(..., ) + idx] = self.adapt_model(self.grid.values[(..., ) + idx], target_resolution, target_wavelength, resolution_model, wav_cont, res_cont, remove_continuum)
        
    def adapt_model(self, model_to_adapt: np.ndarray, target_resolution: np.ndarray, target_wavelength: np.ndarray, resolution_model: np.ndarray, wav_cont: np.ndarray=[], res_cont: np.ndarray=[], remove_continuum: bool=False):
        '''
        Method to adapt a specific model at a given resolution and wavelength grid

        Args
        ----------
        model_to_adapt    (np.ndarray): Model to adapt
        target_resolution      (float): Target resolution for to reach
        target_wavelength (np.ndarray): Target wavelength grid to reah
        resolution_model  (np.ndarray): Resolution of the model interpolated onto the target wavelength grid
        wav_cont          (np.ndarray): Wavelength of the continuum
        res_cont          (np.ndarray): Resolution of the continuum
        remove_continuum        (bool): Whether to remove the continuum
        '''
        
        if len(resolution_model) != len(target_wavelength):
            print('The resolution of the model has not been interpolated to the target wavelength grid')
            raise ForMoSAError()
        
        
        model_spectro = resolution_decreasing(self.wavelength, model_to_adapt, resolution_model, target_wavelength, target_resolution)
        
        # If we want to estimate and substract the continuum of the data (except for high contrast where we need to keeo the og spectrum):
        if remove_continuum == True:
            model_spectro -= continuum_estimate(target_wavelength, model_spectro, target_resolution, wav_cont, res_cont)  
        
        return model_spectro
 
