import numpy as np 
from pathlib import Path
import glob
import os 
import xarray as xr
from ForMoSA.utils_spec import resolution_decreasing, continuum_estimate
from scipy.interpolate import interp1d
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import multiprocessing as mp
from functools import partial
from sklearn.decomposition import PCA
from torchnmf.nmf import NMF
import torch
import ForMoSA.utils as utils


class ForMoSAError(Exception):
    pass

class ModelGrid(object):
    '''
    Model class, which provides easy access to a grid of model
    
    Parameters
    ----------
    path (str | os.PathLike): Path to the grid of model.
    log_level (str): Log level of the handler, by default ``'info'`` for all important informations.
    
    Authors: Allan Denis
    '''
    
    def __init__(self, path: str | os.PathLike, logger) -> None:
        
        # the command .expanduser() transforms the path in a full absolute path, removing any '~' in the path
        self._model_path = Path(path).expanduser()
        self._adapted_grid = dict()
        self._logger = logger
        self._read_grid()
        
    ##################################################
    # Representation
    ##################################################

    def __repr__(self) -> str:
        if len(self.wavelength) > 0:
            return f'<Grid, name={self.name}, root={self.root}, wavelength={min(self.wavelength):.3f}-{max(self.wavelength):.3f}, resolution={min(self.resolution):.3f}-{max(self.resolution):.3f}>'
        else:
            return '<Empty grid>'
        
    def __format__(self) -> str:
        return self.__repr__()

    ##################################################
    # Properties
    ##################################################

    @property  
    def model_path(self):               # Path to the grid
        return self._model_path
    
    @property 
    def root(self):                     # Root of path to the grid
        return self.model_path.parent

    @property 
    def name(self):                     # Name of the grid
        return str(self.model_path).split('/')[-1].split('.nc')[0]
        
    @property  
    def attrs(self):                    # Attritutes of the grid
        return self._attrs
    
    @property  
    def wavelength(self):               # Wavelength of the grid
        return self._wavelength
    
    @property  
    def nyquist(self):                  # Nyquist resolution
        if len(self.wavelength) > 1:
            return self.wavelength / (2 * np.diff(self.wavelength, append=(2*self.wavelength[-1] - self.wavelength[-2])))
        else:
            return self.wavelength
    
    @property  
    def resolution(self):               # Resolution defined as the minimum between the given resolution and the Nyquist resolution
        # Sometimes self.nyquist < 0 (e.g. photometric data with a few points) so we need to make sure to keep the resolution to 0 in this case    
        return np.maximum(np.zeros(len(self.nyquist)), np.minimum(self.nyquist, self._resolution))   
    
    @property  
    def grid(self):                     # Grid
        return self._grid
    
    @property  
    def adapted_grid(self):             # Adapted grid
        return self._adapted_grid
    
    @property 
    def counter(self):                  # Counter for the number of adapted grids : starts from 0
        return len(self.adapted_grid) - 1
    
    @property 
    def keys(self):                     # Keys of the grid parameters
        return self.attrs['key']
    
    @property 
    def titles(self):                   # Names of the grid parameters
        return self.attrs['title']
    
    @property 
    def key_values(self):               # Values taken by the grid parameters
        values = {}
        for key in self.keys:
            values[key] = self.grid[key].values
        return values
    
    @property 
    def valid_spectra(self):            # Boolean where False correspond to grid indices with no spectra
        return ~np.isnan(self.grid).values
    
    @property 
    def lims_params_grid(self):
        return {par : [min(self.key_values[par]), max(self.key_values[par])] for par in self.keys}

    
    ##################################################
    # Methods
    ##################################################
      
  
    def _read_grid(self):
        '''
        Method to read the model grid and store important information
        
        Authors: Allan Denis
        '''
        
        ds = xr.open_dataset(self.model_path, decode_cf=False, engine="netcdf4")
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
        
        Authors: Allan Denis
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
      
        
    def _add_subgrid(self, wavelength_spectro: np.ndarray, resolution_spectro: np.ndarray, wavelength_photo: np.ndarray, ins_photo: np.ndarray = np.array([]), indobs: int = 0, obs_name: str = 'unknown'):
        '''
        Add a subgrid to be adapted to a specific wavelength and a specific resolution
        The subgrid inherits from the Grid class so the same methods can be used for the subgrid

        Parameters
        ----------
        wavelength_spectro (np.ndarray): wavelength grid of the model (spectroscopy)
        resolution_spectro (np.ndarray): resolution grid of the model (spectroscopy)
        wavelength_photo   (np.ndarray): wavelength grid of the model (photometry)
        ins_photo                 (str): Instrument of the photometry point
        obs_name                  (str): Name of the observation
        
        Authors: Allan Denis
        '''
        
        sub_spectro = ModelSubGrid(parent_grid=self, path=self._model_path, logger=self._logger, target_wavelength=wavelength_spectro, target_resolution=resolution_spectro, obs_name=obs_name, component_type = 'spectro')
        sub_photo = ModelSubGrid(parent_grid=self, path=self._model_path, logger=self._logger, target_wavelength=wavelength_photo, target_resolution=np.zeros(len(wavelength_photo)), ins_photo=ins_photo, obs_name=obs_name, component_type = 'photo')
        
        self._adapted_grid[indobs] = {'spectro': sub_spectro, 'photo': sub_photo}
    
        
    def adapt_grid(self, target_resolution: np.ndarray, target_wavelength: np.ndarray, wavelength_photo: np.ndarray = [], ins_photo: np.ndarray = [], wav_cont: np.ndarray = [], res_cont: np.ndarray = [], remove_continuum: bool = False, obs_name: str = 'unknown'):
        '''
        Adapt the grid of models to a given resolution and wavelength.
    
        Parameters
        ----------
        target_resolution (np.float64): Target resolution to reach
        target_wavelength (np.float64): Target wavelength to reach
        remove_continuum (bool): Whether to remove the continuum
        
        Authors: Simon Petrus, Matthieu Ravet, Paulina Palma-Bifani, Arthur Vigan and Allan Denis
        '''
      
        self._logger.info(f' Adapt model {self.grid.name} to the observation {obs_name}')  
      
        if len(ins_photo) > 0:
            self._check_photometry_filters_exist(ins_photo)

        self._logger.debug(f'< Add a subgrid for the observation {obs_name}.>')
        self._add_subgrid(target_wavelength, target_resolution, wavelength_photo, ins_photo, self.counter + 1, obs_name)
        
        interp_mod_to_obs = interp1d(self.wavelength, self.resolution, fill_value='extrapolate')
        resolution_model = interp_mod_to_obs(target_wavelength)
             
        shape = self.grid.values.shape[1:]
        pbar = tqdm(total=np.prod(shape), leave=False)

        def update_result(result, idx, model):
            model._adapted_grid[model.counter]['spectro']._grid[(..., ) + idx] = result[0]
            model._adapted_grid[model.counter]['photo']._grid[(..., ) + idx] = result[1]
            pbar.update(1)
            
        try: # Parallel if possible
            self._logger.info('< Parallel adaptation.>')
            ncpu = mp.cpu_count()
            with ThreadPool(processes=ncpu) as pool:
                for idx in np.ndindex(shape):
                    callback = partial(update_result, idx=idx, model=self)
                    pool.apply_async(self._adapt_model, args=(idx, target_resolution, target_wavelength, wavelength_photo, ins_photo, resolution_model, wav_cont, res_cont, remove_continuum), callback=callback)
                pool.close()
                pool.join()
                
        except Exception as e:
            self._logger.warning(f'< Parallel adaptation produced the following error: {e}. Trying non parallel implementation.')
            try:
                for idx in np.ndindex(shape):
                    result = self._adapt_model(idx, target_resolution, target_wavelength, wavelength_photo, ins_photo, resolution_model, wav_cont, res_cont, remove_continuum)
                    self._adapted_grid[self.counter]['spectro']._grid[(..., ) + idx] = result[0]
                    self._adapted_grid[self.counter]['photo']._grid[(..., ) + idx] = result[1]
            except Exception as e:
                msg = f' Non parallel adaptation produced the following error: {e}.'
                self._logger.critical(msg)
                raise ForMoSAError(msg)
       
        
    def _adapt_model(self, idx: np.ndarray, target_resolution: np.ndarray, target_wavelength: np.ndarray, wavelength_photo: np.ndarray, ins_photo: str, resolution_model: np.ndarray, wav_cont: np.ndarray=[], res_cont: np.ndarray=[], remove_continuum: bool=False):
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
        
        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''
        
        try:
            model_to_adapt = self._load_model_at_specific_index(idx)
            model_spectro = None
            model_photo = None
            
            # Spectroscopy
            if len(resolution_model) > 0 and len(target_wavelength) > 0:
                model_spectro = resolution_decreasing(self.wavelength, model_to_adapt, resolution_model,
                                                      target_wavelength, target_resolution)
                if remove_continuum:
                    model_spectro -= continuum_estimate(target_wavelength, model_spectro,
                                                        target_resolution, wav_cont, res_cont)
            
            # Photometry
            if len(wavelength_photo) > 0 and len(ins_photo) > 0:
                model_photo = np.zeros(len(ins_photo))
                
                # Check that all the filters file exist
                self._check_photometry_filters_exist(ins_photo)
                    
                for pho_ind, pho in enumerate(ins_photo):
                    filter_path = self._find_filter_file(pho)
                    filter_pho = np.load(filter_path)
                    x_filt = filter_pho['x_filt']
                    y_filt = filter_pho['y_filt']
                    filter_interp = interp1d(x_filt, y_filt, fill_value="extrapolate")
                    y_filt_interp = filter_interp(self.wavelength)
        
                    ind = np.where((self.wavelength > min(x_filt)) & (self.wavelength < max(x_filt)))
                    delta_lambda = self.wavelength[ind][1] - self.wavelength[ind][0]
                    num = np.sum(model_to_adapt[ind] * y_filt_interp[ind] * delta_lambda)
                    denom = np.sum(y_filt_interp[ind] * delta_lambda)
                    model_photo[pho_ind] = num / denom if denom != 0 else np.nan
    
        except ForMoSAError as e:   # This line is necessary when we are in a Threapool to stop the execution of the code
            raise e  
    
        return model_spectro, model_photo
    
        
    def _find_filter_file(self, filter_name: str) -> str | None:
        '''
        Find a filter file .npz given a filter name, ignoring lowercase and uppercase letters.
        
        Parameters
        ----------
        filter_name (str): Name of the filter ('Keck_NIRC2_H', 'NACO_Lp', 'MIRI_F1065', ...)
    
        Returns
        -------
        str | None: Path of the file if it exists, None otherwise
        
        Authors: Allan Denis
        '''
        
        path_list = __file__.split("/")[:-1]
        filter_dir = '/'.join(path_list) + '/phototeque/'
        
        for file_path in glob.glob(os.path.join(filter_dir, '*.npz')):
            root = os.path.basename(file_path).split('.')[0]
            if root.lower() == filter_name.lower():
                return file_path
        
        return None
    
    
    def _check_photometry_filters_exist(self, filters: list[str]) -> list[str]:
        '''
        Method to check that photometric filters exist
        
        Parameters
        ----------
        filters (list of str): List of name of filters
 
        
        Authors: Allan Denis
        '''
        
        missing = []
        for filt in filters:
            if self._find_filter_file(filt) is None:
                file_filt = '/'.join(__file__.split("/")[:-1]) + '/phototeque/' + filt + '.npz'
                missing.append(file_filt)
        
        if missing:
            msg = f" Filter files cannot be found : {', '.join(missing)}."
            self._logger.critical(msg)
            raise ForMoSAError(msg)


    def _emulate_with_PCA(self, PCA_comp='NA'):
        '''
        Emulator of the grid spectra with PCA decomposition.
        
        Args:
            PCA_comp (str | int): Number of PCA components to use, or 'NA' for automatic selection (≥99% variance).
        
        Returns:
            tuple: (flx_grid_mean, flx_grid_std, eigenspectra, weights)
            
        Authors: Matthieu Ravet
        '''
        
        grid = self.grid
        flx_grid = grid.to_numpy()
        og_grid = np.copy(flx_grid)
    
        # Reshape and normalize
        flx_grid = flx_grid.reshape(flx_grid.shape[0], -1).T
        nfs = flx_grid.mean(1)
        flx_grid /= nfs[:, np.newaxis]
    
        flx_grid_mean = flx_grid.mean(0)
        flx_grid -= flx_grid_mean
        flx_grid_std = flx_grid.std(0)
        flx_grid /= flx_grid_std
    
        # PCA
        if PCA_comp == 'NA':
            pca = PCA(n_components=0.99, svd_solver="full")
        else:
            pca = PCA(n_components=PCA_comp, svd_solver="full")
    
        pca.fit_transform(flx_grid)
        vectors = pca.components_
    
        # Compute weights
        m = len(vectors)
        M = len(flx_grid)
        ws = np.empty((M * m,))
        for i in range(m):
            for j in range(M):
                ws[i * M + j] = vectors[i].T @ flx_grid[j]
        ws = ws.reshape((pca.n_components_,) + og_grid.shape[1:])
        nfs = nfs.reshape(og_grid.shape[1:])
        weights = np.concatenate((nfs[np.newaxis, :], ws), axis=0)
    
        return flx_grid_mean, flx_grid_std, vectors, weights
    
    
    def _emulate_with_NMF(self, NMF_comp):
        '''
        Emulator of the grid spectra with NMF decomposition (requires CUDA if available).
        
        Args:
            NMF_comp (int): Number of NMF components to use.
        
        Returns:
            tuple: (eigenspectra, weights)
            
        Authors: Matthieu Ravet
        '''
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(0)
    
        grid = self.grid
        flx_grid = grid.to_numpy()
        og_grid = np.copy(flx_grid)
    
        # Reshape
        flx_grid = flx_grid.reshape(flx_grid.shape[0], -1)
    
        flx_grid_torch = torch.tensor(flx_grid, device=device, dtype=torch.float64)
        model = NMF(flx_grid_torch.shape, rank=NMF_comp).to(device)
        model.fit(flx_grid_torch, max_iter=10000)
    
        H = model.H.cpu().detach().numpy()
        W = model.W.cpu().detach().numpy()
    
        vectors = H.T
        weights = W.T.reshape((NMF_comp,) + og_grid.shape[1:])
    
        return vectors, weights
    
    
    def _interpolate_missing_values(self, method: str = "linear", limit: int = None, fill_value: str = 'extrapolate', max_gap: int = None) -> None:
        '''
        Interpolate missing (NaN) values in the adapted spectroscopic and photometric grids.
    
        Parameters
        ----------
        method      (str): Interpolation method to use.
        limit       (int): Maximum number of consecutive NaNs to fill.
        fill_value  (str): Method to fill in points outside of data range
        max_gap     (int): Maximum size of gap, a continuous sequence of NaNs, that will be filled
    
        Authors: Simon Petrus, Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        '''
        
        self._logger.info('Interpolate between holes of the grid')
    
        interp_kwargs = {
            "method": method,
            "fill_value": fill_value,
            "limit": limit,
            "max_gap": max_gap,
        }
    
        def interpolate_component(component, comp_type):
            self._logger.info(f' {component.name} -- {component.obs_name} -- {comp_type.capitalize()} ')
            # Check that grid is a xarray.DataArray
            if not(isinstance(component.grid, xr.DataArray)):
                msg = " Grid is not a valid xarray.DataArray."
                self._logger.error(msg)
                raise ForMoSAError(msg)
            # Check that grid is not empty
            if len(component.wavelength) == 0:
                   msg = ' Empty grid.'
                   self._logger.warning(msg)
            else:       
                for idx, (key, title) in enumerate(zip(self.keys, self.titles)):
                    self._logger.info(f' {idx+1}/{len(self.keys)} - {title}')
                    if component.grid.isnull().any(dim=key).any().item():
                        component._grid = component.grid.interpolate_na(dim=key, **interp_kwargs)
        
        # If self is an instance of ModelSubGrid (e.g. ModelGrid[0]['spectro])
        if isinstance(self, ModelSubGrid):
            interpolate_component(self, self.component_type)
        # If self is an instance of ModelGrid (Recommanded case)
        else:
            for key, grid in self._adapted_grid.items():
                for comp_type in ['spectro', 'photo']:
                    component = grid[comp_type]
                    # Avoid interpolate_component function to produce Empty grid warnings for the grids that do not correspond to the current observation (e.g. 'spectro' grid for photometric observations)
                    if len(component.wavelength) > 0:
                        interpolate_component(component, comp_type)
                        
                        
    def _interpolate_between_gridpoints(self, theta: list, method: str = "linear", indobs: int = 0) -> None:
        '''
        Interpolate between gridpoints in the adapted spectroscopic and photometric grids.
    
        Parameters
        ----------
        theta      (list): List of parameters values to interpolate the models grid to
        method      (str): Interpolation method to use.
        indobs      (int): Index of the observation we want to interpolate
    
        Authors: Simon Petrus, Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        
        Returns
        -------
        grid_interp  (xarray.DataArray): interpolated grid
        '''
        
        self._logger.info('Interpolate between holes of the grid')
        
        if len(theta) != len(self.keys):    # Check if the number of parameters to interpolate the models grid to match the number of grid parameters
            msg = f' The number of parameters ({len(theta)}) to interpolate the grid to does not correspond to the number of grid parameters ({len(self.keys)}).'
            self._logger.critical(msg)
            raise ForMoSAError(msg)
            
        # Building dictionary of interpolation parameters
        interp_kwargs = {}
        for i, name in enumerate(self.keys):
            interp_kwargs[name] = theta[i]
    
        # Adding 'method' and 'fill_value' options
        interp_kwargs['method'] = method
        interp_kwargs['kwargs'] = {'fill_value': 'extrapolate'}
    
        def interpolate_component(component, comp_type, interp_kwargs):
            self._logger.info(f' {component.name} -- {component.obs_name} -- {comp_type.capitalize()} ')
            # Check that grid is a xarray.DataArray
            if not(isinstance(component.grid, xr.DataArray)):
                msg = " Grid is not a valid xarray.DataArray."
                self._logger.error(msg)
                raise ForMoSAError(msg)
            else:       
                grid_interp = component.grid.interp(**interp_kwargs)
        
        # If self is an instance of ModelSubGrid (e.g. ModelGrid[0]['spectro])
        if isinstance(self, ModelSubGrid):
            interpolate_component(self, self.component_type)
        # If self is an instance of ModelGrid (Recommanded case)
        else:
            grid_interp = []
            for comp_type in ['spectro', 'photo']:
                component = self.adapted_grid[indobs][comp_type]
                # Avoid interpolate_component function to produce Empty grid warnings for the grids that do not correspond to the current observation (e.g. 'spectro' grid for photometric observations)
                if not(component.is_empty):
                    grid_interp.append(interpolate_component(component, comp_type, interp_kwargs))
                if component.is_empty:
                    grid_interp += (np.asarray([]),)
           
    
    def _save_grid(self, store_path: str | os.PathLike) -> None:
        '''
        Save the adapted spectroscopic and photometric grids separately to a specified directory.
    
        Parameters
        ----------
        store_path (str | os.PathLike): Path to the directory where adapted grids will be saved.
    
        Authors: Simon Petrus, Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        '''
        
        self._logger.info('Save the adapted grids')
        store_path = Path(store_path).expanduser()
    
        def save_component(component, comp_type, grid_name, obs_name):
            self._logger.info(f' {component.name} -- {component.obs_name} -- {comp_type.capitalize()} ')
            # Check that the grid is a xr.DataArray
            if not(isinstance(component.grid, xr.DataArray)):
                msg = " Grid is not a valid xarray.DataArray."
                self._logger.error(msg)
                raise ForMoSAError(msg)
            else:
                filename = f"adapted_grid_{comp_type}_{grid_name}_{obs_name}_nonan.nc"
                self._logger.debug(f'< Save grid to {store_path / filename} >')
                component.grid.attrs['ins_photo'] = ", ".join(component.ins_photo)
                component.grid.attrs['res'] = component.resolution
                ds = component.grid.to_dataset()
                ds.to_netcdf(store_path / filename, format="NETCDF4", engine="netcdf4", mode="w")
    
        # If self is an instance of ModelSubGrid (e.g. ModelGrid[0]['spectro])
        if isinstance(self, ModelSubGrid):
            save_component(self, self.component_type, self.name, self.obs_name)
        # If self is an instance of ModelGrid (Recommanded case)
        else:
            for key, grid in self._adapted_grid.items():
                grid_name = self.name
                obs_name = grid['spectro'].obs_name  
                for comp_type in ['spectro', 'photo']:
                    component = grid[comp_type]
                    if not(component.is_empty):
                        save_component(component, comp_type, grid_name, obs_name)
            

    def _load_grid_from_files(self, store_path: str | os.PathLike, obs_name: str | list = 'unknown') -> None:  
        '''
        Method to load adapted grid files 

        Parameters
        ----------
        store_path (str | os.PathLike) : Path where the grid data are saved
        
        Authors: Simon Petrus, Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        '''
        
        self._logger.info('Load adapted grid from stored adapted grid files.')
        
        
        store_path = Path(store_path).expanduser()
        grid_files = list(store_path.glob('adapted_grid_*.nc'))
        grid_name = self.name
        
        # Check for the existance of the grid files
        if not grid_files:
            msg = f" No grid file in {store_path}. Ensure file format is adapted_grid_{'spectro/photo'}_{'obs_name'}_nonan.nc"
            self._logger.error(msg)
            raise ForMoSAError(msg)
        
        # Check for the parameter obs_name
        if (isinstance(obs_name, str) and len(grid_files) > 1 and not(obs_name == 'unknown')) or (isinstance(obs_name, list) and len(obs_name) != len(grid_files)):
            msg = f' {len(grid_files)} stored adapted grid files found. Please use a list of {len(grid_files)} names for the parameter obs_name'
            self._logger_error(f' {len(grid_files)} stored adapted grid files found. Please use a list of {len(grid_files)} names for the parameter obs_name')
            raise ForMoSAError(msg) 
    
        def load_component(comp_type, grid_name, obs_name):
            grid_file = store_path / f"adapted_grid_{comp_type}_{grid_name}_{obs_name}_nonan.nc"
            if not grid_file.exists():
                return grid_file, False, None
            self._logger.info(f' {comp_type.capitalize()} -- {grid_name} -- {obs_name}')
            self._logger.debug(f'< Open grid file {grid_file} >')
            grid = xr.open_dataset(grid_file, decode_cf=False, engine='netcdf4')
            return None, True, grid.grid
    
        for indobs in range(len(grid_files)):
            if obs_name == 'unknown':
                self._logger.info(' Unknown observation name. Trying to retrieve it from the stored files name')
                obs_name_indobs = str(grid_files[indobs]).split('_')[-2]
            else:
                obs_name_indobs = obs_name[indobs]
                
            missing_files = []
            for comp_type in ['spectro', 'photo']:
                missing, loaded, grid = load_component(comp_type, grid_name, obs_name_indobs)
                if not loaded:
                    missing_files.append(str(missing))
                else:
                    wavelength, resolution, ins_photo = grid.coords['wavelength'], grid.attrs['res'], grid.attrs['ins_photo']
                    if comp_type == 'spectro':
                        self._add_subgrid(wavelength, resolution, np.array([]), np.array([]), indobs, obs_name_indobs)
                    else:
                        self._add_subgrid(np.array([]), np.array([]), wavelength, ins_photo, indobs, obs_name_indobs)
                        
            if len(missing_files) == 2:
                msg = f"Grid files cannot be found: {', '.join(missing_files)}"
                self._logger.error(msg)
                raise ForMoSAError(msg)
                        
                    
class ModelSubGrid(ModelGrid):
    '''
    Subclass of the class ModelGrid defining a subgrid to be adapted to a specific wavelength and a specific resolution
    
    Parameters
    ----------
    parent_grid        (ModelGrid): Instanciation of the ModelGrid class
    logger                (Logger): Logger
    target_wavelength (np.ndarray): Target wavelength of the subgrid
    target_resolution (np.ndarray): Target resolution of the subgrid
    
    Authors: Allan Denis
    '''
    
    def __init__(self, path: str | os.PathLike, parent_grid: ModelGrid, logger, target_wavelength: np.ndarray, target_resolution: np.ndarray, ins_photo: np.ndarray = np.array([]), obs_name: str = 'unknown', component_type = 'unknown'):
        super().__init__(path, logger)
    
        # Attributes specific to this subgrid
        self._wavelength = target_wavelength
        self._resolution = target_resolution
        self._attrs['res'] = self.resolution
        self._ins_photo = ins_photo
        self._parent_wavelength = parent_grid.wavelength
        self._parent_resolution = parent_grid.resolution
        self._obs_name = obs_name
        self._component_type = component_type
    
        # Shape of the 
        base_shape = parent_grid.grid.shape[1:]
    
        # Initialization of an empty grid 
        grid_shape = (len(target_wavelength),) + base_shape
        empty_grid = np.empty(grid_shape)
        empty_grid[:] = np.nan
        
        coords = {k: parent_grid.grid.coords[k] for k in parent_grid.grid.dims if k != 'wavelength'}
        coords['wavelength'] = target_wavelength

        self._grid = xr.DataArray(data=empty_grid, dims=('wavelength',) + parent_grid.grid.dims[1:], coords=coords, name='grid')
    
    
    ##################################################
    # Properties
    ##################################################
    
    @property 
    def obs_name(self):               # Name of observation
        return self._obs_name
    
    @property 
    def ins_photo(self):              # Name of instrument for photometry
        return self._ins_photo
    
    @property
    def parent_wavelength(self):      # Wavelength of parent grid
        return self._parent_wavelength
    
    @property
    def parent_resolution(self):      # Resolution of parent grid
        return self._parent_resolution
    
    @property 
    def component_type(self):         # Compoent type ('spectro', 'photo')
        if self._component_type == 'unknown':
            if len(self.ins_photo) == 0:
                return 'spectro'
            return 'photo'
        return self._component_type
    
    @property 
    def is_empty(self):
        return len(self.wavelength) == 0
    
