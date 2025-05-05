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

class Grid(object):
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
        self._logger = logger
        self._read_grid()
        
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
    def counter(self):    # Counter for the number of adapted grids : starts from 0
        return len(self.adapted_grid) - 1
    
    @property 
    def keys(self):
        return self.attrs['key']
    
    @property 
    def titles(self):
        return self.attrs['title']
    
    @property 
    def key_values(self):
        values = {}
        for key in self.keys:
            values[key] = self.grid[key].values
        return values
    
    @property 
    def valid_spectra(self):
        return ~np.isnan(self.grid).values

    
    ##################################################
    # Methods
    ##################################################
  
    def _read_grid(self):
        '''
        Method to read the model grid and store important information
        '''
        
        ds = xr.open_dataset(self.path, decode_cf=False, engine="netcdf4")
        self._wavelength = ds['wavelength'].values
        self._resolution = ds.attrs['res']
        self._attrs = ds.attrs
        self._attrs['res'] = self.resolution
        self._grid = ds['grid']
        
    def _load_model_at_specific_index(self, idx: tuple):
        '''
        Method to load a model in the grid at a given index
        
        Parameters
        ----------
        idx (tuple): Index of model to be loaded
        '''
        model_to_return = self.grid.values[(..., ) + idx]
        if np.any(np.isnan(model_to_return)):
            msg = 'Extraction of model failed : '
            for i, (key, title) in enumerate(zip(self.keys, self.titles)):
                msg += f'{title}={self.key_values[key][idx[i]]}, '
            self._logger.warning(f' {msg}')
            return None
        else:
            return model_to_return
        
    def _add_subgrid(self, target_wavelength: np.ndarray, target_resolution: np.ndarray):
        '''
        Add a grid to be adapted to a specific wavelength and a specific resolution

        Parameters
        ----------
        indobs (int): Observation number
        wavelength (np.ndarray): wavelength grid of the model
        resolution (np.ndarray): resolution grid of the model
        grid (np.float64): grid of the model
        '''
        sub = SubGrid(parent_grid=self, logger=self._logger, target_wavelength=target_wavelength, target_resolution=target_resolution)
        self._adapted_grid[self.counter+1] = sub
    
        
    def adapt_grid(self, target_resolution: np.ndarray, target_wavelength: np.ndarray, wav_cont: np.ndarray=[], res_cont: np.ndarray=[], remove_continuum: bool=False):
        '''
        Adapt the grid of models to a given resolution and wavelength.
    
        Parameters
        ----------
        target_resolution (np.float64): Target resolution to reach
        target_wavelength (np.float64): Target wavelength to reach
        remove_continuum (bool): Whether to remove the continuum
        '''
    
        self._add_subgrid(target_wavelength, target_resolution)
        
        interp_mod_to_obs = interp1d(self.wavelength, self.resolution, fill_value='extrapolate')
        resolution_model = interp_mod_to_obs(target_wavelength)
             
        shape = self.grid.values.shape[1:]
        pbar = tqdm(total=np.prod(shape), leave=False)

        def update_result(result, idx, model):
            model._adapted_grid[model.counter]._grid[(..., ) + idx] = result
            pbar.update(1)
            
        try: # Parallel if possible
            ncpu = mp.cpu_count()
            with ThreadPool(processes=ncpu) as pool:
                for idx in np.ndindex(shape):
                    callback = partial(update_result, idx=idx, model=self)
                    pool.apply_async(self.adapt_model, args=(idx, target_resolution, target_wavelength, resolution_model, wav_cont, res_cont, remove_continuum), callback=callback)
                pool.close()
                pool.join()
                
        except:
            for idx in np.ndindex(shape):
                self._adapted_grid.grid[self.counter].values[(..., ) + idx] = self.adapt_model(idx, target_resolution, target_wavelength, resolution_model, wav_cont, res_cont, remove_continuum)
        
        
    def adapt_model(self, idx: np.ndarray, target_resolution: np.ndarray, target_wavelength: np.ndarray, resolution_model: np.ndarray, wav_cont: np.ndarray=[], res_cont: np.ndarray=[], remove_continuum: bool=False):
        '''
        Method to adapt a specific model at a given resolution and wavelength grid

        Args
        ----------
        idx               (np.ndarray): Index of model to adapt
        target_resolution      (float): Target resolution for to reach
        target_wavelength (np.ndarray): Target wavelength grid to reah
        resolution_model  (np.ndarray): Resolution of the model interpolated onto the target wavelength grid
        wav_cont          (np.ndarray): Wavelength of the continuum
        res_cont          (np.ndarray): Resolution of the continuum
        remove_continuum        (bool): Whether to remove the continuum
        '''
        
        if len(resolution_model) != len(target_wavelength):
            self._logger.critical(' The resolution of the model has not been interpolated to the target wavelength grid')
            raise ForMoSAError()
        
        # Retrieve model from index
        model_to_adapt = self._load_model_at_specific_index(idx)
        
        if np.any(np.isnan(model_to_adapt)):
            msg = 'Extraction of model failed : '
            for i, (key, title) in enumerate(zip(self.keys, self.titles)):
                msg += f'{title}={self.key_values[key][idx[i]]}, '
            self._logger.warning(f' {msg}')
        
        model_spectro = resolution_decreasing(self.wavelength, model_to_adapt, resolution_model, target_wavelength, target_resolution)
        
        # If we want to estimate and substract the continuum of the data (except for high contrast where we need to keeo the og spectrum):
        if remove_continuum == True:
            model_spectro -= continuum_estimate(target_wavelength, model_spectro, target_resolution, wav_cont, res_cont)  
        
        return model_spectro
    
    
    def _generate_xarray_from_grid(self, target_resolution: np.ndarray, target_wavelength: np.ndarray):
        return
 

class SubGrid(Grid):      
    def __init__(self, parent_grid: Grid, logger, target_wavelength: np.ndarray, target_resolution: np.ndarray):
        # Copy of attritutes of principal model
        # Logger et identifiants
        self._logger = logger
        self._root = parent_grid.root
        self._name = parent_grid.name
        self._attrs = parent_grid.attrs.copy()

        # Valeurs spectrales spécifiques à cette sous-grille
        self._wavelength = target_wavelength
        self._resolution = target_resolution
        self._attrs['res'] = self.resolution

        # Shape of the 
        base_shape = parent_grid.grid.shape[1:]

        # Initialisation d’une grille vide pour cette résolution
        grid_shape = (len(target_wavelength),) + base_shape
        empty_grid = np.empty(grid_shape)
        empty_grid[:] = np.nan
        self._grid = empty_grid
        
    @property
    def grid(self):
        return self._grid

