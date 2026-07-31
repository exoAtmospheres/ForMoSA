import logging
import numpy as np
from tqdm import tqdm

import ForMoSA.utils.spec as us
from ForMoSA.core.errors import ForMoSAError
from ForMoSA.grid.model_grid import ModelGrid
from ForMoSA.core.loggings import setup_logging
from ForMoSA.core.enums import LogLikelihoodType
from ForMoSA.transform.observed import ObservedModel
from ForMoSA.utils.misc import get_weighted_percentile
from ForMoSA.core.enums import ParameterKind, ObservationKeys
from ForMoSA.nested_sampling.nested_sampling import NestedSampling
from ForMoSA.transform.apply_effects import ApplyObservationEffects

class NSAnalysis(object):
    '''
    Class used to subsequent analysis of nested sampling products.
    It includes the reconstructions of models, best fit, computations of ccf, ...

    Parameters
    ----------
    ns : NestedSampling
        Instance of class NestedSampling
    logger : logging.Logger
        Logger
    log_level : str
        Level of the Logger

    Notes
    -----
    Authors: Allan Denis
    '''

    def __init__(self, ns: NestedSampling, logger: logging.Logger | None = None, log_level: str = 'INFO') -> None:

        self._logger = logger if logger is not None else setup_logging(level=log_level, name='NSAnalysis')
        self._ns = ns
        self._validate()

    # ===================
    # Properties
    # ===================

    @property
    def logger(self) -> logging.Logger:
        """Logger."""
        return self._logger

    @property
    def ns(self) -> NestedSampling:
        """Instance of NestedSampling."""
        return self._ns

    @property
    def best_fit(self) -> list[ObservedModel]:
        """Best fit."""
        if self.ns.results is None:
            raise ForMoSAError('Please first run the Nested Sampling before computing the best fit', self.logger)

        best_params = list(self.ns.results.median_parameters.values())
        return self.build_models_from_theta(best_params)

    @property
    def native_best_fit(self) -> ObservedModel:
        """Best fit parameters applied to the native model."""
        if self.ns.results is None:
            raise ForMoSAError('Please first run the Nested Sampling before computing the best fit', self.logger)

        # Best parameters
        best_params = list(self.ns.results.median_parameters.values())

        # wavelength range
        wrange = (self.ns.restricted_observations.wavelength_range[0] * 0.95, self.ns.restricted_observations.wavelength_range[1] * 1.05)

        # build restricted grid
        grid = self.ns.subgrids.parent_grid._restricted_grid(f'{wrange[0]},{wrange[1]}', print_logger=True)
        # Interpolate nan values
        grid._interpolate_missing_values()

        return self._build_native_observed_model(grid, best_params, print_logger=True)

    # ===================
    # Methods
    # ===================

    def _validate(self) -> None:
        '''
        Validation for NSAnalysis.

        Notes
        -----
        Authors: Allan Denis
        '''

        if not isinstance(self.ns, NestedSampling):
            raise ForMoSAError(f'Wrong type for ns: {type(self.ns)}. Expeceted an instance of NestedSampling')

    def build_models_from_theta(self, theta):
        '''
        Build the models from the values of the free parameters drawn by the Nested Sampling

        Parameters
        ----------
        theta : np.ndarray[float]
            List of values picked by the Nested Sampling for the free parameters

        Returns
        -------
        list[ObservedModel]
            List of instances of class ObservedModel

        Notes
        -----
        Authors: Allan Denis
        '''

        return self.ns.build_models_from_theta(theta)

    def _build_native_observed_model(self, grid: ModelGrid, params: list[float], print_logger: bool = False) -> ObservedModel:
        '''
        Build native observed model including analytic scaling.

        Parameters
        ----------
        grid : ModelGrid
            Instance of class ModelGrid
        params : list[float]
            list or array model parameters
        print_logger : bool
            Whether to print the logger

        Returns
        -------
        ObservedModel
            Native observed model

        Notes
        -----
        Authors: Allan Denis
        '''

        if not isinstance(grid, ModelGrid):
            raise ForMoSAError(f'Wrong type for grid: {type(grid)}. Expected an instance of ModelGrid', self.logger)

        # Build observed parameters
        observed_params = self.ns._build_params_for_obs(params, 0).global_params

        # Build native model from parameters
        native_model = ObservedModel.from_grid_and_params(grid, observed_params)

        # If radius is fitted, no analytic scaling needed
        if observed_params.has_kind(ParameterKind.R):
            return native_model

        # Radius is not fitted, so we need to analytically scale the model
        # In order to do that, we need to concatenate all adapted models and all adapted observations to make them comparable

        wave_all = np.array([])
        flux_all = np.array([])
        res_all = np.array([])

        ind_obs_list = []
        for i in range(self.ns.observations.n_observations):
            # Retrieve adapted subgrid and obs
            subgrid = self.ns.restricted_subgrids.subgrids[i]
            obs = self.ns.restricted_observations.observations[i]

            # Build instance of ObservedModel from parameters
            observed = ObservedModel.from_grid_and_params(subgrid, observed_params)

            if len(observed.wave) != len(obs.wave):
                observed = ApplyObservationEffects._apply_resolution_decreasing(observed, obs)

            if not obs.hc_mode:
                # Concatenate everything
                wave_all = np.append(wave_all, observed.wave)
                flux_all = np.append(flux_all, observed.flux)
                res_all = np.append(res_all, observed.res)

                ind_obs_list.append(i)

        # Build stacked observed_model and observations
        stacked_model = ObservedModel(wave_all, flux_all, res_all)
        stacked_model._sort()
        stacked_obs = self.ns.restricted_observations._stack(ind_obs_list, print_logger=print_logger)

        if np.any(np.isnan(stacked_model.flux)):
            return ObservedModel(native_model.wave, [np.nan] * len(native_model.wave), native_model.res)

        # analytic scaling
        stacked_model.flux, ck = us.calc_ck(
            stacked_model.flux,
            stacked_obs[ObservationKeys.FLUX.canonical],
            stacked_obs[ObservationKeys.ERROR.canonical],
            0,
            0,
            analytic='yes',
        )

        native_model.flux *= ck

        return native_model

    def best_fit_intervals(self, percs: list[float] = (0.68,)) -> list[tuple[ObservedModel, ObservedModel]]:
        '''
        Confidence intervals of the native best fit, for one or several credible levels at once.

        All requested levels share the same (expensive) pass over the posterior samples, so
        requesting several levels here is much cheaper than calling `best_fit_interval` once
        per level.

        Parameters
        ----------
        percs : list[float]
            Percentile values between 0 and 1 (0.68 for 1 sigma, 0.95 for 2 sigmas, ...)

        Returns
        -------
        list[tuple[ObservedModel, ObservedModel]]
            For each requested percentile, the lower and higher envelope of the flux.
            All returned models share the same wavelength grid: the subset of the native
            model's wavelength grid that is defined for every posterior sample (RV shifts
            trim a few points at the edges differently for each sample; those edge points
            are dropped here rather than mixed with NaNs).

        Notes
        -----
        Authors: Allan Denis
        '''

        percs = [float(p) for p in percs]

        # Initial checks
        if self.ns.results is None:
            raise ForMoSAError('Please first run the Nested Sampling before computing the best fit', self.logger)

        for p in percs:
            if p < 0 or p > 1:
                raise ForMoSAError(f'perc must be a float between 0 and 1. Got {p} with type {type(p)}', self.logger)

        # wavelength range
        wrange = (self.ns.observations.wavelength_range[0] * 0.95, self.ns.observations.wavelength_range[1] * 1.05)

        # build restricted grid
        grid = self.ns.subgrids.parent_grid._restricted_grid(f'{wrange[0]},{wrange[1]}', print_logger=True)
        # Interpolate nan values
        grid._interpolate_missing_values()

        # Fixed reference wavelength grid shared by every sample: `_build_native_observed_model`
        # always builds its native model from this same `grid`, and RV Doppler-shifting only
        # ever drops points from it (it never resamples), so every sample's wavelengths are an
        # exact index-subset of `ref_wave`. This lets us align samples by index instead of
        # interpolating them onto a common grid.
        ref_wave = grid.wave
        samples = self.ns.results.samples[self.ns.results.burn_in:]
        weights = self.ns.results.weights[self.ns.results.burn_in:]

        self.logger.info(f'    Computing confidence interval(s) for percentile(s) {np.round(percs, 2).tolist()}')

        models_flux = np.full((len(samples), len(ref_wave)), np.nan)
        res_ref = np.full(len(ref_wave), np.nan)
        for i, sample in enumerate(tqdm(samples)):
            observed_model = self._build_native_observed_model(grid, sample, print_logger=False)
            idx = np.searchsorted(ref_wave, observed_model.wave)
            models_flux[i, idx] = observed_model.flux
            res_ref[idx] = observed_model.res

        # Only keep wavelength points present for every sample (RV shifts trim a handful of
        # points at the edges, differently per sample); get_weighted_percentile is not
        # NaN-aware, so mixing in the missing edges would silently bias the percentiles there.
        valid = ~np.any(np.isnan(models_flux), axis=0)
        wave_valid = ref_wave[valid]
        res_valid = res_ref[valid]
        models_flux = models_flux[:, valid]

        # Compute every requested percentile in a single vectorized call
        n_requested = np.concatenate([[(1 - p) / 2 * 100, (1 + p) / 2 * 100] for p in percs])
        percentiles = get_weighted_percentile(n_requested, models_flux, weights=weights)

        return [
            (ObservedModel(wave_valid, percentiles[2 * i], res_valid), ObservedModel(wave_valid, percentiles[2 * i + 1], res_valid))
            for i in range(len(percs))
        ]

    def best_fit_interval(self, perc: float = 0.68) -> tuple[ObservedModel, ObservedModel]:
        '''
        Confidence interval of the native best fit.

        Parameters
        ----------
        perc : float
            Percentile value between 0 and 1 (0.68 for 1 sigma, 0.95 for 2 sigmas)

        Returns
        -------
        tuple[ObservedModel, ObservedModel]
            lower and higher values of the flux for the confidence interval

        Notes
        -----
        Authors: Allan Denis
        '''

        return self.best_fit_intervals([perc])[0]

    def compute_ccf(self, rv_grid: np.ndarray, index: int = 0, theta: list | None = None, logL_type: LogLikelihoodType = LogLikelihoodType.CHI2) -> dict[str, np.ndarray]:
        '''
        Compute and optionally plot the Cross-Correlation Function (CCF).

        Parameters
        ----------
        rv_grid : np.ndarray
            Grid of radial velocity values (in km/s)
        index : int
            Index of observation used for the ccf computation
        theta : list
            List of free values of the parameters. If not provided, the best fitted parameters are used
        logL_type : LogLikelihoodType
            Type of log-likelihood used

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary of CCF results keyed by observation name

        Notes
        -----
        Authors: Bhavesh Rajpoot and Allan Denis
        '''

        if theta is not None:
            if len(theta) != len(self.ns.parameters.free_names):
                raise ForMoSAError(f'If you provide a list of free values theta, make sure it has the same length of the number of free parameters {len(self.ns.parameters.free_names)}', self.logger)

        results = {}

        obs = self.ns.restricted_observations.observations[index]
        subgrid = self.ns.restricted_subgrids.subgrids[index]

        if not obs.is_spectroscopic:
            raise ForMoSAError(f'observation {obs.name} is not spectroscopic. You cannot compute the CCF')

        if theta is None:
            theta = list(self.ns.results.median_parameters.values())

        observed_params = self.ns._build_params_for_obs(theta, index)

        if not observed_params.has_kind(ParameterKind.RV):
            raise ForMoSAError('You must provide a RV parameter to compute rv/vsini map', self.logger)

        # Make sure RV parameter is set to 0
        for p in observed_params.values:
            if p.kind == ParameterKind.RV:
                observed_params.values[p] = 0

        native_model = ObservedModel.from_grid_and_params(subgrid, observed_params)

        wav_fit = self.ns.wave_fit[index]

        star_flx = obs.star_flux if obs.star_flux is not None else np.array([])
        transm = obs.transm if obs.transm is not None else np.array([])
        system = obs.system if obs.system is not None else np.array([])

        file_tag = f'{obs.name}_{index}'

        self._logger.info(f'      Computing RV CCF for observation {obs.name}')

        ccf_raw, acf, ccf_star, rv_peak, logL, ccf = us.compute_ccf(
            native_model.wave,
            native_model.flux,
            obs.wave,
            obs.flux,
            obs.err,
            native_model.res,
            obs.res,
            subgrid.res_cont,
            wav_fit,
            star_flx_obs_spectro=star_flx,
            transm_obs_spectro=transm,
            system_obs_spectro=system,
            rv_grid=rv_grid,
            rv_sini_map=False,
            logL_type=logL_type
        )

        # Find best RV
        best_idx = np.unravel_index(np.argmax(ccf), ccf.shape)
        best_rv = rv_grid[best_idx[0]]

        self._logger.info(f'      Best RV = {best_rv:.1f} km/s')

        results[file_tag] = {
            'rv_grid': rv_grid,
            'ccf': ccf,
            'acf': acf,
            'ccf_star': ccf_star,
            'rv_peak': rv_peak,
            'logL': logL,
            'ccf_raw': ccf_raw
        }

        return results

    def compute_rv_vsini_map(self, rv_grid: np.ndarray, vsini_grid: np.ndarray, index: int = 0, theta: list | None = None, logL_type: LogLikelihoodType = LogLikelihoodType.CHI2) -> dict[str, np.ndarray]:
        '''
        Compute and optionally plot the RV vs v.sin(i) loglikelihood map.

        Parameters
        ----------
        rv_grid : np.ndarray
            Grid of radial velocity values (in km/s)
        vsini_grid : np.ndarray
            Grid of v.sin(i) values (in km/s)
        index : int
            Index of observation used for the ccf computation
        theta : list
            List of free values of the parameters. If not provided, the best fitted parameters are used
        logL_type : LogLikelihoodType 
            Type of log-likelihood used

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary of RV-vsini map results keyed by observation name

        Notes
        -----
        Authors: Bhavesh Rajpoot and Allan Denis
        '''

        if theta is not None:
            if len(theta) != len(self.ns.parameters.free_names):
                raise ForMoSAError(f'If you provide a list of free values theta, make sure it has the same length of the number of free parameters {len(self.ns.parameters.free_names)}', self.logger)

        results = {}

        obs = self.ns.restricted_observations.observations[index]
        subgrid = self.ns.restricted_subgrids.subgrids[index]

        if not obs.is_spectroscopic:
            raise ForMoSAError(f'observation {obs.name} is not spectroscopic. You cannot compute the CCF')

        if theta is None:
            theta = list(self.ns.results.median_parameters.values())

        observed_params = self.ns._build_params_for_obs(theta, index)

        # Make sure RV parameter is set to 0
        for p in observed_params.values:
            if p.kind == ParameterKind.RV:
                observed_params.values[p] = 0

        # Make sure VSINI parameter is set to 0
        for p in observed_params.values:
            if p.kind == ParameterKind.VSINI:
                observed_params.values[p] = 0
                vsini_type = p.vsini_function.value

        if not observed_params.has_kind(ParameterKind.RV):
            raise ForMoSAError('You must provide a RV parameter to compute rv/vsini map', self.logger)

        if not observed_params.has_kind(ParameterKind.VSINI):
            raise ForMoSAError('You must provide a VSINI parameter to compute rv/vsini map', self.logger)

        native_model = ObservedModel.from_grid_and_params(subgrid, observed_params)

        ld_value = observed_params.get_kind(ParameterKind.LD)

        wav_fit = self.ns.wave_fit[index]

        star_flx = obs.star_flux if obs.star_flux is not None else np.array([])
        transm = obs.transm if obs.transm is not None else np.array([])
        system = obs.system if obs.system is not None else np.array([])

        file_tag = f'{obs.name}_{index}'

        self._logger.info(f'      Computing RV-vsini map for observation {file_tag}')

        logL_map = np.zeros((len(vsini_grid), len(rv_grid)))

        for j, vsini_val in enumerate(tqdm(vsini_grid, desc=f'RV-vsini map ({file_tag})', leave=False)):
            flx_broadened, res_broadened = us.vsini_fct(
                native_model.wave, native_model.flux, native_model.res, ld_value, vsini_val, vsini_type
            )

            logL = us.compute_ccf(
                native_model.wave,
                flx_broadened,
                obs.wave,
                obs.flux,
                obs.err,
                res_broadened,
                obs.res,
                subgrid.res_cont,
                wav_fit,
                star_flx_obs_spectro=star_flx,
                transm_obs_spectro=transm,
                system_obs_spectro=system,
                rv_grid=rv_grid,
                rv_sini_map=True,
                logL_type=logL_type
            )

            logL_map[j] = logL

        # Find best RV and vsini
        best_idx = np.unravel_index(np.argmax(logL_map), logL_map.shape)
        best_vsini = vsini_grid[best_idx[0]]
        best_rv = rv_grid[best_idx[1]]

        self._logger.info(f'      Best RV = {best_rv:.1f} km/s, best vsini = {best_vsini:.1f} km/s')

        results[file_tag] = {
            'rv_grid': rv_grid,
            'vsini_grid': vsini_grid,
            'logL_map': logL_map,
            'best_rv': best_rv,
            'best_vsini': best_vsini
        }

        return results


