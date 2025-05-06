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
        self._root = Path(path).expanduser().parent
        self._name = str(Path(path).expanduser()).split('/')[-1].split('.nc')[0]
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
        if len(self.wavelength) > 1:
            return self.wavelength / (2 * np.diff(self.wavelength, append=(2*self.wavelength[-1] - self.wavelength[-2])))
        else:
            return self.wavelength
    
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
    def titles(self):        # Names of the grid parameters
        return self.attrs['title']
    
    @property 
    def key_values(self):    # Values taken by the grid parameters
        values = {}
        for key in self.keys:
            values[key] = self.grid[key].values
        return values
    
    @property 
    def valid_spectra(self):   # Boolean where False correspond to grid indices with no spectra
        return ~np.isnan(self.grid).values

    
    ##################################################
    # Methods
    ##################################################
      
  
    def _read_grid(self):
        '''
        Method to read the model grid and store important information
        
        Authors: Allan Denis
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
        
    def _add_subgrid(self, wavelength_spectro: np.ndarray, resolution_spectro: np.ndarray, wavelength_photo: np.ndarray, ins_photo: str):
        '''
        Add a subgrid to be adapted to a specific wavelength and a specific resolution
        The subgrid inherits from the Grid class so the same methods can be used for the subgrid

        Parameters
        ----------
        wavelength_spectro (np.ndarray): wavelength grid of the model (spectroscopy)
        resolution_spectro (np.ndarray): resolution grid of the model (spectroscopy)
        wavelength_photo   (np.ndarray): wavelength grid of the model (photometry)
        ins_photo                 (str): Instrument of the photometry point
        
        Authors: Allan Denis
        '''
        
        sub_spectro = ModelSubGrid(parent_grid=self, logger=self._logger, target_wavelength=wavelength_spectro, target_resolution=resolution_spectro)
        sub_photo = ModelSubGrid(parent_grid=self, logger=self._logger, target_wavelength=wavelength_photo, target_resolution=np.zeros(len(wavelength_photo)))
        
        self._adapted_grid[self.counter+1] = {'spectro': sub_spectro, 'photo': sub_photo, 'ins_photo': ins_photo}
    
        
    def adapt_grid(self, target_resolution: np.ndarray = [], target_wavelength: np.ndarray = [], wavelength_photo: np.ndarray = [], ins_photo: np.ndarray = [], wav_cont: np.ndarray = [], res_cont: np.ndarray = [], remove_continuum: bool = False):
        '''
        Adapt the grid of models to a given resolution and wavelength.
    
        Parameters
        ----------
        target_resolution (np.float64): Target resolution to reach
        target_wavelength (np.float64): Target wavelength to reach
        remove_continuum (bool): Whether to remove the continuum
        
        Authors: Simon Petrus, Matthieu Ravet, Paulina Palma-Bifani, Arthur Vigan and Allan Denis
        '''
      
        if len(ins_photo) > 0:
            missing_filters = self._check_photometry_filters_exist(ins_photo)
            if missing_filters:
                msg = f" Filter files cannot be found : {', '.join(missing_filters)}"
                self._logger.critical(msg)
                raise ForMoSAError(msg)

        self._add_subgrid(target_wavelength, target_resolution, wavelength_photo, ins_photo)
        
        interp_mod_to_obs = interp1d(self.wavelength, self.resolution, fill_value='extrapolate')
        resolution_model = interp_mod_to_obs(target_wavelength)
             
        shape = self.grid.values.shape[1:]
        pbar = tqdm(total=np.prod(shape), leave=False)

        def update_result(result, idx, model):
            model._adapted_grid[model.counter]['spectro']._grid[(..., ) + idx] = result[0]
            model._adapted_grid[model.counter]['photo']._grid[(..., ) + idx] = result[1]
            pbar.update(1)
            
        try: # Parallel if possible
            ncpu = mp.cpu_count()
            with ThreadPool(processes=ncpu) as pool:
                for idx in np.ndindex(shape):
                    callback = partial(update_result, idx=idx, model=self)
                    pool.apply_async(self.adapt_model, args=(idx, target_resolution, target_wavelength, wavelength_photo, ins_photo, resolution_model, wav_cont, res_cont, remove_continuum), callback=callback)
                pool.close()
                pool.join()
                
        except Exception:
            for idx in np.ndindex(shape):
                self._adapted_grid.grid[self.counter]['spectro']._grid[(..., ) + idx] = self.adapt_model(idx, target_resolution, target_wavelength, resolution_model, wav_cont, res_cont, remove_continuum)
        
        
    def adapt_model(self, idx: np.ndarray, target_resolution: np.ndarray, target_wavelength: np.ndarray, wavelength_photo: np.ndarray, ins_photo: str, resolution_model: np.ndarray, wav_cont: np.ndarray=[], res_cont: np.ndarray=[], remove_continuum: bool=False):
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
        
            if model_to_adapt is None or np.any(np.isnan(model_to_adapt)):
                msg = 'Extraction of model failed : '
                for i, (key, title) in enumerate(zip(self.keys, self.titles)):
                    msg += f'{title}={self.key_values[key][idx[i]]}, '
                self._logger.warning(f'{msg}')
                return None, None
            
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
    
        except ForMoSAError as e:  # When we are in a ThreadPool, this line is necessary to make the thread stop the execution of the code
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
        Check that photometric filters exist
        
        Parameters
        ----------
        filters (list of str): List of name of filters
    
        Returns
        -------
        list of str: List of lacking filters
        
        Authors: Allan Denis
        '''
        
        missing = []
        for filt in filters:
            if self._find_filter_file(filt) is None:
                missing.append(filt)
        
        return missing


    def _emulate_with_PCA(self, PCA_comp='NA'):
        '''
        Emulator of the grid spectra with PCA decomposition.
        
        Args:
            PCA_comp (str | int): Number of PCA components to use, or 'NA' for automatic selection (≥99% variance).
        
        Returns:
            tuple: (flx_grid_mean, flx_grid_std, eigenspectra, weights)
            
        Authors: Mathieu Ravet
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
    
    
    def emulate_with_NMF(self, NMF_comp):
        '''
        Emulator of the grid spectra with NMF decomposition (requires CUDA if available).
        
        Args:
            NMF_comp (int): Number of NMF components to use.
        
        Returns:
            tuple: (eigenspectra, weights)
            
        Authors: Mathieu Ravet
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

    
    
    
class ModelSubGrid(ModelGrid):
    '''
    Subclass of the class ModelGrid defining a subgrid to be adapted to a specific wavelength and a specific resolution
    
    Parameters
    ----------
    parent_grid        (ModelGrid): Instanciation of the ModelGrid class
    logger                (Logger): Logger
    target_wavelength (np.ndarray): Target wavelength of the subgrid
    target_resolution (np.ndarray): Target resolution of the subgrid
    '''
    
    def __init__(self, parent_grid: ModelGrid, logger, target_wavelength: np.ndarray, target_resolution: np.ndarray):
        # Copy of attritutes ModelGrid
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
    
        # Initialization of an empty grid 
        grid_shape = (len(target_wavelength),) + base_shape
        empty_grid = np.empty(grid_shape)
        empty_grid[:] = np.nan
        
        coords = {k: parent_grid.grid.coords[k] for k in parent_grid.grid.dims if k != 'wavelength'}
        coords['wavelength'] = target_wavelength

        self._grid = xr.DataArray(data=empty_grid, dims=('wavelength',) + parent_grid.grid.dims[1:], coords=coords, name='grid')
    

