import numpy as np
import os
import xarray as xr
import multiprocessing as mp

from pathlib import Path
from scipy.interpolate import interp1d
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from functools import partial
from sklearn.decomposition import PCA

import ForMoSA.utils.misc as utils

from ForMoSA.utils.spec import resolution_decreasing, continuum_estimate
from ForMoSA.nested_sampling.parameters import NestedSamplingParameters
from ForMoSA.error import ForMoSAError

try:
    import torch
    from torchnmf.nmf import NMF
except ImportError:
    torch = None
    NMF = None


class ModelGrid(object):
    '''
    Model class, which provides easy access to a grid of model

    Parameters
    ----------
    path (str | os.PathLike): Path to the grid of model.
    log_level (str): Log level of the handler, by default ``'info'`` for all important informations.

    Authors: Allan Denis
    '''

    def __init__(self, model_path: str | os.PathLike, logger) -> None:

        # the command .expanduser() transforms the path in a full absolute path, removing any '~' in the path
        self._model_path = Path(model_path).expanduser()
        self._adapted_grid = dict()
        self._logger = logger

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
    def logger(self):                   # Logger
        return self._logger

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

    @grid.setter
    def grid(self, grid_array):         # Grid setter
        self._grid = grid_array
        self._wavelength = grid_array.coords['wavelength'].values
        self._resolution = grid_array.attrs['res']
        self._attrs = grid_array.attrs
        return grid_array

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
    def lims_params_grid(self):         # Limits of grid parameters
        return {par : [min(self.key_values[par]), max(self.key_values[par])] for par in self.keys}

    @property
    def wavelength_range(self):         # Global wavelength range of observations accounting for photometry and spectroscopy

        if self.counter == -1:
            msg = 'No adapted grid loaded. Please load or build at least one adapted grid.'
            self._logger.error(msg)
            raise ForMoSAError(msg)

        wavelengths = []

        for i in range(self.counter + 1):

            # -- Spectroscopy wavelengths
            wavelengths.extend(self.adapted_grid[i]['spectro'].wavelength)

            # -- Photometry wavelengths
            wavelengths.extend(self.adapted_grid[i]['photo'].wavelength)

            # -- Check if photometry filters exist
            ins_photo = np.atleast_1d(self.adapted_grid[i]['photo'].instrument)

            if len(ins_photo) > 0 and not np.all(ins_photo == 'unknown'):
                self._check_photometry_filters_exist(ins_photo)

                for pho in ins_photo:
                    filter_path = utils.find_filter_file(pho)
                    filter_pho = np.load(filter_path)
                    x_filt = filter_pho['x_filt']

                    # Append min and max of the filter bandpass
                    wavelengths.extend([np.min(x_filt), np.max(x_filt)])

        # Convert to numpy for convenience
        wavelengths = np.asarray(wavelengths)

        return (wavelengths.min(), wavelengths.max())


    ##################################################
    # Methods
    ##################################################

    def _read_grid(self):
        '''
        Method to read the model grid and store important information

        Authors: Allan Denis
        '''

        self._logger.debug(f'< Read grid in {self.model_path}')
        ds = xr.open_dataset(self.model_path, decode_cf=False, engine="netcdf4")
        self._wavelength = ds['wavelength'].values
        self._resolution = ds.attrs['res']
        self._attrs = ds.attrs
        ds.attrs['res'] = self.resolution    # self.resolution returns the minimum between self._resolution and the Nyquist sampled resolution
        self._grid = ds['grid']
        self.grid.attrs = self.attrs


    def _find_valid_resolution_region(self, wavelength: np.ndarray, resolution: np.ndarray, target_wavelength: np.ndarray, target_resolution: np.ndarray | float) -> np.ndarray:
        '''
        Method to identify the best continuous region of resolution > 0 covering a part of the target grid

        Args:
            wavelength               (array): Grid of wavelength
            resolution               (array): Associated resolution
            target_wavelength        (array): Target wavelength grid
            target_resolution  (array|float): Associated target resolution

        Returns:
            best_indices (np.ndarray): indices of the best part

        Authors: Allan Denis
        '''

        increasing = np.insert(np.diff(wavelength) > 0, 0, True)  # True is wavelength[i+1] > wavelength[i]

        edges = np.diff(increasing.astype(int))
        starts = np.where(np.insert(edges == 1, 0, increasing[0]))[0]
        ends = np.where(np.append(edges == -1, increasing[-1]))[0] + 1

        candidate_regions = []
        for start, end in zip(starts, ends):
            wl_region = wavelength[start:end]
            if wl_region[-1] < target_wavelength[0] or wl_region[0] > target_wavelength[-1]:
                continue
            overlap_mask = (target_wavelength[0] <= wl_region) & (wl_region <= target_wavelength[-1])
            if not np.any(overlap_mask):
                continue
            sub_start = start + np.where(overlap_mask)[0][0] - 1
            sub_end = start + np.where(overlap_mask)[0][-1] + 2
            sub_start, sub_end = max(0, sub_start), min(len(wl_region), sub_end)
            res_metric = np.min(resolution[sub_start:sub_end])
            candidate_regions.append((sub_start, sub_end, res_metric))

        max_res = (np.max(target_resolution) if isinstance(target_resolution, np.ndarray) else target_resolution)

        best_diff = np.inf
        best_indices = None
        for sub_start, sub_end, res_metric in candidate_regions:
            if res_metric >= max_res:
                diff = abs(res_metric - max_res)
                if diff < best_diff:
                    best_diff = diff
                    best_indices = np.arange(sub_start, sub_end)

        if best_indices is None:
            best_indices = np.arange(len(wavelength))

        return best_indices


    def _determine_target_wavelength_and_resolution(self, wav_obs_spectro: np.ndarray, res_obs_spectro: np.ndarray, target_res_mod: str | float, params: NestedSamplingParameters, indobs: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Determine target wavleength and resolution and filter model

        Args:
            wav_obs_spectro                    (array): Wavelength grid of data
            res_obs_spectro                    (array): Resolution of data
            target_res_mod               (str | float): Target resolution of model to reach ('obs', 'mod' or float)
            params          (NestedSamplingParameters): Instance of :class:~NestedSamplingParameters
            indobs                               (int): Index of observation

        Returns:
            target_wavelength     (array): Target wavelength to reach for the model
            target_resolution     (array): Target resolution to reach for the model
            resolution_interp     (array): Resolution of the model interpolated onto target_wavelength

        Authors: Allan Denis
        """

        if target_res_mod == 'mod':
            target_wavelength, target_resolution = self.wavelength, self.resolution
        elif target_res_mod == 'obs':
            target_wavelength, target_resolution = wav_obs_spectro, res_obs_spectro
        else:
            target_wavelength = self.wavelength
            target_resolution = np.full(len(self.wavelength), float(target_res_mod))

        if len(wav_obs_spectro) > 0:
            # Additional step to account for the possible presence of rv priors
            margin = 0.01 if ('rv' in params.parameters.keys()) or (f'rv_{indobs}' in params.parameters.keys()) else 0.0

            if target_res_mod != 'obs':
                wmin, wmax = wav_obs_spectro[0], wav_obs_spectro[-1]
                mask_mod_obs = ((target_wavelength >= wmin * (1 - margin)) & (target_wavelength <= wmax * (1 + margin)) & (target_resolution >= max(res_obs_spectro)))
                target_wavelength = target_wavelength[mask_mod_obs]
                target_resolution = target_resolution[mask_mod_obs]

                # Find wavelength corresponding to the observation accounting for possible overlaps in the target_wavelength grid
                best_indices = self._find_valid_resolution_region(target_wavelength, target_resolution, target_wavelength, target_resolution)
                # Removing overlaps in target_wavelength and associated indices in target_resolution
                target_wavelength = target_wavelength[best_indices]
                target_resolution = target_resolution[best_indices]

            # Same step with the raw wavelength and resolution of the grid
            _, wavelength_grid, resolution_grid = self._select_wavelength_range(self.grid, target_wavelength, target_resolution)
        else:
            resolution_grid = self.resolution

        return target_wavelength, target_resolution, resolution_grid



    def _select_wavelength_range(self, model: xr.DataArray, target_wavelength: np.ndarray, target_resolution: np.ndarray) -> tuple[xr.DataArray, np.ndarray, np.ndarray]:
        '''
        Filter the model according to the target grid accounting for overlaping wavelengths

        Args:
            model             (xr.DataArray): Model we want to filter
            target_wavelength        (array): Target wavelength to reach for the model
            target_resolution        (array): Target resolution to reach for the model

        Returns:
            filtered_model        (xr.DataArray)
            selected_wavelengths  (np.ndarray)
            selected_resolutions  (np.ndarray)

        Authors: Allan Denis
        '''

        wavelength = self.wavelength
        resolution = self.resolution

        best_indices = self._find_valid_resolution_region(wavelength, resolution, target_wavelength, target_resolution)

        filtered_model = model.isel(wavelength=best_indices)
        filtered_model = filtered_model.assign_coords(wavelength=wavelength[best_indices])
        filtered_model = filtered_model.assign_attrs(res=resolution[best_indices])

        return filtered_model, wavelength[best_indices], resolution[best_indices]


    def _load_model_at_specific_index(self, idx: tuple):
        '''
        Method to load a model in the grid at a given index

        Parameters
        ----------
        idx (tuple): Index of model to be loaded (e.g. (5,0,6,3) for a 4 parameters grid)

        Authors: Allan Denis
        '''

        model_to_return = self.grid[(..., ) + idx]
        if np.any(np.isnan(model_to_return)):
            msg = 'Extraction of model failed : '
            for i, (key, title) in enumerate(zip(self.keys, self.titles)):
                msg += f'{title}={self.key_values[key][idx[i]]}, '
            self._logger.warning(f' {msg}')
            return np.asarray([])
        else:
            return model_to_return


    def _add_subgrid(self, wavelength_spectro: np.ndarray, resolution_spectro: np.ndarray, wavelength_photo: np.ndarray, ins_spectro: str = 'unknown', ins_photo: str = 'unknown', indobs: int = 0, obs_name: str = 'unknown', grid_spectro: xr.DataArray = None, grid_photo: xr.DataArray = None):
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
        grid             (xr.DataArray): Grid

        Authors: Allan Denis
        '''

        sub_spectro = ModelSubGrid(parent_grid=self.grid, path=self._model_path, logger=self._logger, target_wavelength=wavelength_spectro, target_resolution=resolution_spectro, ins=ins_spectro, obs_name=obs_name, component_type = 'spectro', grid = grid_spectro)
        sub_photo = ModelSubGrid(parent_grid=self.grid, path=self._model_path, logger=self._logger, target_wavelength=wavelength_photo, target_resolution=np.zeros(len(wavelength_photo)), ins=ins_photo, obs_name=obs_name, component_type = 'photo', grid = grid_photo)

        self._adapted_grid[indobs] = {'spectro': sub_spectro, 'photo': sub_photo}


    def adapt_all_grids(self, obs_data: dict, target_res_mod: list, parameters: NestedSamplingParameters, wav_cont: list=['NA'], res_cont: list=['NA']):
        '''
        Method to adapt the grid of models to each observation present in obs_data

        Parameters
        ----------
        obs_data                       (dict): Dictionary of observations {indobs: {'spectro': spectro_data}, {'photo': photo_data}}
        target_res_mod                 (list): Target resolution of the model to reach for each observation (['obs' | 'mod' | float, ...])
        parameters    (NestedSampling_Params): Instance of :clacc:'~NestedSampling_Params'
        res_cont                 (list): Resolution of the continuum of each of the adapted grid ([float | 'NA', ...])
        wav_cont                 (list): Wavelength grid of the continuum of each of the adapted grid ([np.ndarray | 'NA', ...])

        Authors: Allan Denis
        '''

        for indobs in range(len(obs_data)):
            remove_continuum = False
            # Determine whether to remove the continuum
            if res_cont[indobs % len(res_cont)] != 'NA':
                # High-contrast mode, we do not remove the continuum to the models
                if len(obs_data[indobs]['spectro']['star_flx']) > 0:
                    remove_continuum = False
                # Non high_contrast mode, we remove the continuum to the models
                else:
                    remove_continuum = True

            # Launch adaptation of grid
            self._adapt_grid(obs_data[indobs]['spectro']['wav'], obs_data[indobs]['spectro']['res'], obs_data[indobs]['photo']['wav'], target_res_mod[indobs % len(target_res_mod)], parameters, obs_data[indobs]['spectro']['ins'], obs_data[indobs]['photo']['ins'], wav_cont = wav_cont[indobs % len(wav_cont)], res_cont = res_cont[indobs % len(res_cont)], remove_continuum=remove_continuum, obs_name = obs_data[indobs]['obs_name'])


    def _adapt_grid(self, wav_obs_spectro: np.ndarray, res_obs_spectro: np.ndarray, wav_obs_photo: np.ndarray, target_res_mod: str | float, params: NestedSamplingParameters, ins_spectro: np.ndarray = np.array([]), ins_photo: np.ndarray = np.array([]), wav_cont: np.ndarray | str = 'NA', res_cont: float | str = 'NA', remove_continuum: bool = False, obs_name: str = 'unknown'):
        '''
        Adapt the grid of models to a given resolution and wavelength.

        Parameters
        ----------
        target_resolution_spectro      (array): Target resolution to reach for the spectroscopic adapted grid
        target_wavelength_spectro      (array): Target wavelength to reach for the spectroscopic adapted grid
        target_wavelength_photo        (array): Target wavelength to reach for the photometric adapted grid
        target_res_mod           (str | float): Target resolution to reach for the model ('obs', 'mod', or float)
        ins_spectro                    (array): Spectroscopic instrument
        ins_photo                      (array): Photometric instrument
        wav_cont                 (array | str): Wavelength grid of the continuum estimation
        res_cont                 (array | str): Resolution of the continuum
        remove_continuum                (bool): Whether to remove the continuum
        obs_name                         (str): Name of the observation associated to the adapted grid

        Authors: Simon Petrus, Matthieu Ravet, Paulina Palma-Bifani, Arthur Vigan and Allan Denis
        '''

        self._logger.info(f' Adapt model {self.name} to the observation {obs_name}')

        target_wavelength_spectro, target_resolution_spectro, resolution_model = self._determine_target_wavelength_and_resolution(wav_obs_spectro, res_obs_spectro, target_res_mod, params)
        if len(wav_obs_photo) > 0:
            self._check_photometry_filters_exist(ins_photo)

        if remove_continuum == True and res_cont == 'NA':
            msg = " If you want to remove the continuum, please set values for 'wav_cont' and 'res_cont'."
            self._logger.error(msg)
            raise ForMoSAError(msg)
        if remove_continuum == False and res_cont != 'NA':
            self._logger.warning(" You do not want to remove the continuum but 'res_cont' is defined. Ignoring it and chosing not to remove the continuum.")

        self._logger.debug(f'< Add a subgrid for the observation {obs_name}.>')
        self._add_subgrid(target_wavelength_spectro, target_resolution_spectro, wav_obs_photo, ins_spectro = ins_spectro, ins_photo = ins_photo, indobs = self.counter + 1, obs_name = obs_name)


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
                    model_to_adapt = self._load_model_at_specific_index(idx)
                    pool.apply_async(self._adapt_model, args=(model_to_adapt, target_resolution_spectro, target_wavelength_spectro, wav_obs_photo, ins_photo, resolution_model, wav_cont, res_cont, remove_continuum), callback=callback)
                pool.close()
                pool.join()

        except Exception as e:
            self._logger.warning(f'< Parallel adaptation produced the following error: {e}. Trying non parallel implementation.')
            try:
                for idx in tqdm(np.ndindex(shape)):
                    model_to_adapt = self._load_model_at_specific_index(idx)
                    result = self._adapt_model(model_to_adapt, target_resolution_spectro, target_wavelength_spectro, wav_obs_photo, ins_photo, resolution_model, wav_cont, res_cont, remove_continuum)
                    self._adapted_grid[self.counter]['spectro']._grid[(..., ) + idx] = result[0]
                    self._adapted_grid[self.counter]['photo']._grid[(..., ) + idx] = result[1]
            except Exception as e:
                msg = f' Non parallel adaptation produced the following error: {e}.'
                self._logger.critical(msg)
                raise ForMoSAError(msg)


    def _adapt_model(self, model_to_adapt, target_resolution: np.ndarray, target_wavelength: np.ndarray, wavelength_photo: np.ndarray, ins_photo: np.ndarray, resolution_model: np.ndarray, wav_cont: np.ndarray=[], res_cont: np.ndarray=[], remove_continuum: bool=False):
        '''
        Method to adapt a specific model at a given resolution and wavelength grid

        Args
        ----------
        model_to_adapt         (array): Model to adapt
        target_resolution      (float): Target resolution to reach for the spectroscopic adapted grid
        target_wavelength      (array): Target wavelength grid to reah for the spectroscopic adapted grid
        wavelength_photo       (array): Target wavelength to reach for the photometric adapted grid
        ins_photo              (array): List of photometric instrument names
        resolution_model       (array): Resolution of the model
        wav_cont               (array): Wavelength of the continuum
        res_cont               (array): Resolution of the continuum
        remove_continuum        (bool): Whether to remove the continuum

        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''

        try:
            model_spectro = np.empty(len(target_wavelength))
            model_photo = np.empty(len(wavelength_photo))

            if len(model_to_adapt) > 0:
                # Spectroscopy
                if len(resolution_model) > 0 and len(target_wavelength) > 0:
                    model_to_adapt, _, _ = self._select_wavelength_range(model_to_adapt, target_wavelength, target_resolution)
                    model_spectro = resolution_decreasing(model_to_adapt.coords['wavelength'].values, model_to_adapt.values, resolution_model, target_wavelength, target_resolution)

                    if remove_continuum:
                        model_spectro -= continuum_estimate(target_wavelength, model_spectro,
                                                            target_resolution, wav_cont, res_cont)

                # Photometry
                if len(wavelength_photo) > 0 and len(ins_photo) > 0:
                    model_photo = np.zeros(len(ins_photo))
                    self._check_photometry_filters_exist(ins_photo)

                    for pho_ind, pho in enumerate(ins_photo):
                        filter_path = utils.find_filter_file(pho)
                        filter_pho = np.load(filter_path)
                        x_filt = filter_pho['x_filt']
                        y_filt = filter_pho['y_filt']
                        cut = y_filt > 1e-2
                        x_filt, y_filt = x_filt[cut], y_filt[cut]

                        # Sekect useful part of the spectrum
                        model_sub, _, _ = self._select_wavelength_range(model_to_adapt, x_filt, np.zeros(len(x_filt)))

                        wl_sub = model_sub.coords["wavelength"].values
                        flux_sub = model_sub.values

                        # Interpolate filter on current spectral grid
                        filter_interp = interp1d(x_filt, y_filt, kind="linear", bounds_error=False, fill_value=0)
                        y_filt_interp = filter_interp(wl_sub)

                        # Spectral width
                        delta_lambda = np.diff(wl_sub)
                        delta_lambda = np.append(delta_lambda, delta_lambda[-1])  # Last point must have a weights

                        # Integrate model on filter bandwidth
                        num = np.sum(flux_sub * y_filt_interp * delta_lambda)
                        denom = np.sum(y_filt_interp * delta_lambda)

                        model_photo[pho_ind] = num / denom if denom != 0 else np.nan


        except ForMoSAError as e:   # This line is necessary when we are in a Threapool to stop the execution of the code
            raise e

        return model_spectro, model_photo


    def _check_photometry_filters_exist(self, filters: str | list[str]) -> list[str]:
        '''
        Method to check that photometric filters exist

        Parameters
        ----------
        filters (str | list of str): List of name of filters


        Authors: Allan Denis
        '''

        filters = np.atleast_1d(filters)
        missing = []
        for filt in filters:
            if utils.find_filter_file(filt) is None:
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
        if torch is None or NMF is None:
            msg = "NMF decomposition requires the 'torch' and 'torchnmf' packages. Please install them to use this feature."
            self._logger.error(msg)
            raise ForMoSAError(msg)

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


    def _interpolate_between_gridpoints(self, theta: list, method: str = "linear", indobs: int = 0, print_logger: bool = False) -> list | xr.DataArray:
        '''
        Interpolate between gridpoints in the adapted spectroscopic and photometric grids.

        Parameters
        ----------
        theta         (list): List of parameters values to interpolate the models grid to
        method         (str): Interpolation method to use.
        indobs         (int): Index of the observation we want to interpolate
        print_logger  (bool): Whether to prin the information of the logger
        custom        (bool): Whether to build a custom adapted grid

        Authors: Simon Petrus, Paulina Palma-Bifani, Matthieu Ravet and Allan Denis

        Returns
        -------
            grid_interp  (xarray.DataArray | list): interpolated grid or list containing interpolated grids for spectroscopic and photometric models
        '''

        if print_logger:
            self._logger.info('Interpolate between gridpoints in the grid')

        if len(theta) != len(self.keys):    # Check if the number of parameters to interpolate the models grid to match the number of grid parameters
            msg = f' The number of parameters ({len(theta)}) to interpolate the grid to does not correspond to the number of grid parameters ({len(self.keys)}).'
            self._logger.critical(msg)
            raise ForMoSAError(msg)

        out_of_bounds_params = []

        for i, thetai in enumerate(theta):
            param_name = self.keys[i]
            param_limits = list(self.lims_params_grid.values())[i]
            min_val, max_val = param_limits

            if thetai < min_val or thetai > max_val:
                out_of_bounds_params.append({
                    "param": param_name,
                    "value": thetai,
                    "min": min_val,
                    "max": max_val
                })

        if out_of_bounds_params:
            msg = f"The following parameters are outside the allowed ranges: {out_of_bounds_params}"
            self._logger.critical(msg)
            raise ForMoSAError(msg)


        # Building dictionary of interpolation parameters
        interp_kwargs = {}
        for i, name in enumerate(self.keys):
            # xarray expects plain Python scalars as interpolation indexers.
            # Numpy scalar objects (np.float64, etc.) may raise TypeError.
            value = theta[i]
            if isinstance(value, np.generic):
                value = value.item()
            interp_kwargs[name] = value

        # Adding 'method' and 'fill_value' options
        interp_kwargs['method'] = method
        interp_kwargs['kwargs'] = {'fill_value': np.nan}

        def interpolate_component(component, comp_type, interp_kwargs):
            if print_logger:
                self._logger.info(f' {component.name} -- {component.obs_name} -- {comp_type.capitalize()} ')
            # Check that grid is a xarray.DataArray
            if not(isinstance(component.grid, xr.DataArray)):
                msg = " Grid is not a valid xarray.DataArray."
                self._logger.error(msg)
                raise ForMoSAError(msg)
            else:
                grid_interp = component.grid.interp(**interp_kwargs)
                return grid_interp

        # If self is an instance of ModelSubGrid (e.g. ModelGrid[0]['spectro])
        if isinstance(self, ModelSubGrid):
            grid_interp = interpolate_component(self, self.component_type, interp_kwargs)

        # If self is an instance of ModelGrid
        else:
            comp_type = 'nativ grid'
            grid_interp = interpolate_component(self, comp_type, interp_kwargs)

        return grid_interp


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
                component.grid.attrs['ins'] = ", ".join(component.instrument)
                component.grid.attrs['res'] = component.resolution
                component.grid.to_netcdf(store_path / filename, format="NETCDF4", engine="netcdf4", mode="w")

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
            self._logger.error(f' {len(grid_files)} stored adapted grid files found. Please use a list of {len(grid_files)} names for the parameter obs_name')
            raise ForMoSAError(msg)

        def load_component(comp_type, grid_name, obs_name):
            grid_file = store_path / f"adapted_grid_{comp_type}_{grid_name}_{obs_name}_nonan.nc"
            if not grid_file.exists():
                return grid_file, False, None
            self._logger.info(f' {comp_type.capitalize()} -- {grid_name} -- {obs_name}')
            self._logger.debug(f'< Open grid file {grid_file} >')
            grid = xr.open_dataset(grid_file, decode_cf=False, engine='netcdf4')
            grid = grid['grid']
            return None, True, grid

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
                    wavelength, resolution, ins = grid.coords['wavelength'].values, grid.attrs['res'], grid.attrs['ins']
                    if comp_type == 'spectro':
                        self._add_subgrid(wavelength, resolution, np.array([]), ins_spectro = ins, indobs = indobs, obs_name = obs_name_indobs, grid_spectro = grid)
                    else:
                        self._add_subgrid(np.array([]), np.array([]), wavelength, ins_photo = ins, indobs = indobs, obs_name = obs_name_indobs, grid_photo = grid)

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

    def __init__(self, path: str | os.PathLike, parent_grid: xr.DataArray, logger, target_wavelength: np.ndarray, target_resolution: np.ndarray, component_type: str, ins: str = 'unknown', obs_name: str = 'unknown', grid: xr.DataArray = None):
        super().__init__(path, logger)

        # Attributes specific to this subgrid
        self._wavelength = target_wavelength
        self._resolution = target_resolution
        self._instrument = ins
        self._parent_wavelength = parent_grid.coords['wavelength']
        self._parent_resolution = parent_grid.attrs['res']
        self._obs_name = obs_name
        self._component_type = component_type

        if not(isinstance(grid, xr.DataArray)):
            # Initialization of an empty grid
            base_shape = parent_grid.shape[1:]
            grid_shape = (len(target_wavelength),) + base_shape
            empty_grid = np.empty(grid_shape)
            empty_grid[:] = np.nan

            coords = {k: parent_grid.coords[k] for k in parent_grid.dims if k != 'wavelength'}
            coords['wavelength'] = target_wavelength

            self._grid = xr.DataArray(data=empty_grid, dims=('wavelength',) + parent_grid.dims[1:], coords=coords, name='grid')
            self._attrs = parent_grid.attrs
            self._grid.attrs['res'] = parent_grid.attrs['res']
            self._grid.attrs['ins'] = ins
        else:
            self._grid = grid
            self._attrs = parent_grid.attrs
            self._attrs['res'] = self.resolution
            self._attrs['ins'] = ins

    ##################################################
    # Properties
    ##################################################

    @property
    def obs_name(self):               # Name of observation
        return self._obs_name

    @property
    def instrument(self):             # Name of instrument
        return self._instrument

    @property
    def parent_wavelength(self):      # Wavelength of parent grid
        return self._parent_wavelength

    @property
    def parent_resolution(self):      # Resolution of parent grid
        return self._parent_resolution

    @property
    def component_type(self):         # Compoent type ('spectro', 'photo')
        return self._component_type

    @property
    def is_empty(self):               # Whether the grid is empty
        return len(self.wavelength) == 0

    @property
    def grid(self):
        return self._grid

