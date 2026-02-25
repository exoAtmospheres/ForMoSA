import logging
import numpy as np
from tqdm import tqdm

import ForMoSA.utils.spec as us
from ForMoSA.core.enums import ParameterKind
from ForMoSA.core.errors import ForMoSAError
from ForMoSA.core.loggings import setup_logging
from ForMoSA.transform.observed import ObservedModel
from ForMoSA.utils.misc import get_weighted_percentile
from ForMoSA.nested_sampling.nested_sampling import NestedSampling

class NSAnalysis(object):
    '''
    Class used to subsequent analysis of nested sampling products.
    It includes the reconstructions of models, best fit, computations of ccf, ...

    Parameters
    ----------
    ns            (NestedSampling): Instance of class NestedSampling
    observations  (ObservationSet): Instance of class ObservationSet
    subgrids          (SubGridSet): Instance of class SubGridSet
    logger        (logging.Logger): Logger
    log_level                (str): Level of the Logger

    Authors: Allan Denis
    '''

    def __init__(self, ns: NestedSampling, logger: logging.Logger | None = None, log_level: str = 'INFO') -> None:

        self._logger = logger or setup_logging()
        self._ns = ns
        self._validate()

    # ===================
    # Properties
    # ===================

    @property
    def logger(self) -> logging.Logger:                  # Logger
        return self._logger

    @property
    def ns(self) -> NestedSampling:                      # Instance of NestedSampling
        return self._ns

    @property
    def best_fit(self) -> list[ObservedModel]:            # Best fit
        if self.ns.results is None:
            raise ForMoSAError('Please first run the Nested Sampling before computing the best fit', self.logger)

        best_params = list(self.ns.results.median_parameters.values())
        return self.build_models_from_theta(best_params)

    @property
    def native_best_fit(self) -> ObservedModel:           # Best fit parameters applied to the native model
        if self.ns.results is None:
            raise ForMoSAError('Please first run the Nested Sampling before computing the best fit', self.logger)

        best_params = list(self.ns.results.median_parameters.values())
        wrange = (self.ns.observations.wavelength_range[0]*0.95, self.ns.observations.wavelength_range[1]*1.05)

        grid = self.ns.subgrids.parent_grid._restricted_grid(f'{wrange[0]},{wrange[1]}', print_logger=True)
        grid._interpolate_missing_values()

        observed_params = self.ns._build_params_for_obs(best_params, 0).global_params

        return ObservedModel.from_grid_and_params(grid, observed_params)

    # ===================
    # Methods
    # ===================

    def _validate(self) -> None:
        '''
        Validation for NSAnalysis.

        Authors: Allan Denis
        '''

        if not isinstance(self.ns, NestedSampling):
            raise ForMoSAError(f'Wrong type for ns: {type(self.ns)}. Expeceted an instance of NestedSampling')

    def build_models_from_theta(self, theta):
        '''
        Build the models from the values of the free parameters drawn by the Nested Sampling

        Parameters
        ----------
        theta (np.ndarray[float]): List of values picked by the Nested Sampling for the free parameters

        Returns
        -------
        list[ObservedModel]: List of instances of class ObservedModel

        Authors: Allan Denis
        '''

        return self.ns.build_models_from_theta(theta)

    def best_fit_interval(self, perc: float = 0.68) -> tuple[ObservedModel, ObservedModel]:
        '''
        Confidence interval of the native best fit.

        Parameters
        ----------
        perc (float): Percentile value between 0 and 1 (0.68 for 1 sigma, 0.95 for 2 sigmas)


        Returns
        -------
        tuple[ObservedModel, ObservedModel]: lower and higher values of the flux for the confidence interval

        Authors: Allan Denis
        '''

        perc = float(perc)

        # Initial checks
        if self.ns.results is None:
            raise ForMoSAError('Please first run the Nested Sampling before computing the best fit', self.logger)

        if perc < 0 or perc > 1:
            raise ForMoSAError(f'perc must be a float between 0 and 1. Got {perc} with type {type(perc)}', self.logger)

        lower = (1 - perc) / 2
        upper = (1 + perc) / 2

        wrange = (self.ns.restricted_observations.wavelength_range[0]*0.95, self.ns.restricted_observations.wavelength_range[1]*1.05)
        grid = self.ns.restricted_subgrids.parent_grid._restricted_grid(f'{wrange[0]},{wrange[1]}', print_logger=True)

        models_flux = []

        self.logger.info(f'    Computing confidence interval with percentiles [{np.round(lower,2)} - {np.round(upper,2)}]')
        for sample in tqdm(self.ns.results.samples):
            observed_params = self.ns._build_params_for_obs(sample, 0).global_params

            observed_model = ObservedModel.from_grid_and_params(grid, observed_params)

            models_flux.append(observed_model.flux)

        models_flux = np.array(models_flux)

        perc_1sigma_lower = get_weighted_percentile(lower, models_flux, weights=self.ns.results.weights)
        perc_1sigma_higher = get_weighted_percentile(upper, models_flux, weights=self.ns.results.weights)

        return ObservedModel(observed_model.wave, perc_1sigma_lower, observed_model.res), ObservedModel(observed_model.wave, perc_1sigma_higher, observed_model.res)

    def compute_ccf(self, rv_grid: np.ndarray, index: int = 0, theta: list | None = None) -> dict[str, np.ndarray]:
        '''
        Compute and optionally plot the Cross-Correlation Function (CCF).

        Parameters
        ----------
        rv_grid  (np.ndarray): Grid of radial velocity values (in km/s)
        index           (int): Index of observation used for the ccf computation
        theta          (list): List of free values of the parameters. If not provided, the best fitted parameters are used

        Returns
        -------
        dict[str, np.ndarray]: Dictionary of CCF results keyed by observation name

        Authors: Bhavesh Rajpoot and Allan Denis
        '''

        if theta is not None:
            if len(theta) != len(self.ns.parameters.free_names):
                raise ForMoSAError(f'If you provide a list of free values theta, make sure it has the same length of the number of free parameters {len(self.ns.parameters.free_names)}', self.logger)

        results = {}

        obs = self.ns.restricted_observations.observations[index]
        subgrid = self.ns.restricted_subgrids.subgrids[index]

        if not obs.is_spectroscopic:
            raise ForMoSAError('observation {obs.name} is not spectroscopic. You cannot compute the CCF')

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

        ccf, acf, ccf_star, rv_peak, logL, ccf_raw = us.compute_ccf(
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
            normalize=True
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
            'logL': logL
        }

        return results

    def compute_rv_vsini_map(self, rv_grid: np.ndarray, vsini_grid: np.ndarray, index: int = 0, theta: list | None = None) -> dict[str, np.ndarray]:
        '''
        Compute and optionally plot the RV vs v.sin(i) loglikelihood map.

        Parameters
        ----------
        rv_grid           (np.ndarray): Grid of radial velocity values (in km/s)
        vsini_grid        (np.ndarray): Grid of v.sin(i) values (in km/s)
        index                    (int): Index of observation used for the ccf computation
        theta                  (list): List of free values of the parameters. If not provided, the best fitted parameters are used

        Returns
        -------
        dict[str, np.ndarray]: Dictionary of RV-vsini map results keyed by observation name

        Authors: Bhavesh Rajpoot and Allan Denis
        '''

        if theta is not None:
            if len(theta) != len(self.ns.parameters.free_names):
                raise ForMoSAError(f'If you provide a list of free values theta, make sure it has the same length of the number of free parameters {len(self.ns.parameters.free_names)}', self.logger)

        results = {}

        obs = self.ns.restricted_observations.observations[index]
        subgrid = self.ns.restricted_subgrids.subgrids[index]

        if not obs.is_spectroscopic:
            raise ForMoSAError('observation {obs.name} is not spectroscopic. You cannot compute the CCF')

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
                normalize=False
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


