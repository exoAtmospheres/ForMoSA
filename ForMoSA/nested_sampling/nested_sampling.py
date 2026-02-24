import os
import time
import json
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from ForMoSA.utils.misc import get_weighted_percentile

from ForMoSA.core.errors import ForMoSAError
from ForMoSA.grid.subgrid_set import SubGridSet
from ForMoSA.core.loggings import setup_logging
from ForMoSA.config.global_config import Config_NS
from ForMoSA.nested_sampling.results import NSResults
from ForMoSA.parameter.parameter_set import ParameterSet
from ForMoSA.observation.observation_set import ObservationSet
from ForMoSA.core.enums import NestedAlgorithm, LogLikelihoodType
from ForMoSA.transform.observed import ObservedParameters, ObservedModel

# optional imports for nested sampling algorithms
try:
    import nestle
except ImportError:
    nestle = None

try:
    import pymultinest
except ImportError:
    pymultinest = None

try:
    import ultranest
except ImportError:
    ultranest = None

class NestedSampling(object):
    '''
    ForMoSA Nested_Sampling class, which provides easy access to the parameters of the nested sampling algorithm

    Parameters
    ----------

    logger                         (Logger): Logger
    algorithm                         (str): Algorithm used for the nested sampling ('nestle', 'ultranest' or 'pymultinest')
    npoints                           (int): Number of living points used for the nested sampling
    logL_type     (list[LogLikelihoodType]): List of loglikelihood types used in the nested sampling
    config_NS                   (Config_NS): Instance of class Config_NS containing parameters of ns algorithms for each of the algorithm
    interp_method                     (str): Interpolation method ('linear', 'cubic', 'spline', ...)
    wave_fit                    (list[str]): List of wavelength grid used for fitting
    bounds_lsq                (list[float]): List of bounds used for the least squares (used for high contrast)
    logger                 (logging.Logger): Logger
    log_level                         (str): Level of the Logger

    Authors: Allan Denis
    '''

    def __init__(self, algorithm: NestedAlgorithm, npoints: int, logL_type: list[LogLikelihoodType], config_NS: Config_NS, observations: ObservationSet, subgrids: SubGridSet, parameters: ParameterSet, wave_fit: list[str] | None = None, interp_method: str = 'linear', bounds_lsq: list[tuple[float, float]] | None = None, logger: logging.Logger | None=None, log_level: str='INFO'):
        self._logger = logger or setup_logging(log_level, name='NestedSampling')
        self._logL_type = logL_type

        self._algorithm = algorithm.algo
        self._npoints = npoints
        self._parameters = parameters
        self._observations = observations
        self._subgrids = subgrids
        self._config_NS = config_NS
        self._interp_method = interp_method
        self._wave_fit = wave_fit
        self._bounds_lsq = bounds_lsq
        self._restricted_subgrids = SubGridSet(self._subgrids.parent_grid, logger=self._logger)
        self._restricted_observations = ObservationSet(logger=self._logger)
        self._results = None
        self._validate()
        self._restricted_models_and_data()

    # =================
    # Representation
    # =================

    def __repr__(self):
        return f'NestedSampling(algorithm = {self.algorithm}, npoints = {self.npoints}, status = {"fitted" if not self.results is None else "Not fitted"})'

    # =================
    # Properties
    # =================

    @property
    def algorithm(self) -> str:                             # Algorithm
        return self._algorithm

    @property
    def logL_type(self) -> list:                            # logL functions
        return self._logL_type

    @property
    def observations(self) -> ObservationSet:               # Set of observations
        return self._observations

    @property
    def subgrids(self) -> SubGridSet:                       # Set of subgrids
        return self._subgrids

    @property
    def parameters(self) -> ParameterSet:                   # Priors parameters
        return self._parameters

    @property
    def config_NS(self) -> Config_NS:                       # Config_NS
        return self._config_NS

    @property
    def interp_method(self) -> str:                         # Interpolation method
        return self._interp_method

    @property
    def wave_fit(self) -> str:                              # Wavelengths for fitting
        return self._wave_fit

    @property
    def bounds_lsq(self) -> list[float]:                    # Bounds for the Least Squares estimation
        return self._bounds_lsq

    @property
    def ns_params(self) -> dict:                            # Dictionary of NS algo
        return getattr(self.config_NS, self.algorithm.lower()).to_dict

    @property
    def npoints(self) -> int:                               # Nested sampling number of living points
        return self._npoints

    @property
    def restricted_observations(self) -> ObservationSet:    # Set of restricted observations according to wave_fit
        return self._restricted_observations

    @property
    def restricted_subgrids(self) -> SubGridSet:            # Set of restricted subgrids according to wave_fit
        return self._restricted_subgrids

    @property
    def results(self) -> "NSResults":                       # Results
        return self._results

    @property
    def logger(self) -> logging.Logger:                     # Logger
        return self._logger

    @property
    def to_dict(self) -> dict:                              # Dictionary representation of NestedSampling
        return {
            'algorithm': self.algorithm,
            'npoints': self.npoints,
            'logLtype': [logL.loglike for logL in self.logL_type],
            'config_NS': self.config_NS.to_dict,
            'parameters': self.parameters.to_dict,
            'wave_fit': self.wave_fit,
            'interp_method': self.interp_method,
            'bounds_lsq': [(str(lower), str(upper)) for lower, upper in self.bounds_lsq]
            }

    @property
    def best_fit(self) -> list[ObservedModel]:        # Best fit
        if self.results is None:
            raise ForMoSAError('Please first run the Nested Sampling before computing the best fit', self.logger)

        best_params = list(self.results.median_parameters.values())
        return self.build_models_from_theta(best_params)

    @property
    def native_best_fit(self) -> ObservedModel:       # Best fit parameters applied to the native model
        if self.results is None:
            raise ForMoSAError('Please first run the Nested Sampling before computing the best fit', self.logger)

        median, _ = self.best_fit_interval(perc=0)
        return median

    # =================
    # Class method
    # =================

    @classmethod
    def from_dict(cls, data: dict, observations: ObservationSet, subgrids: SubGridSet, logger: logging.Logger | None=None, log_level: str='INFO') -> 'NestedSampling':
        '''
        Reconstruct an instance of NestedSampling from a dictionary of NestedSampling.

        Parameters
        ----------
        data                   (dict): Dictionary representation NestedSampling parameters
        observations (ObservationSet): Instance of ObservationSet
        subgrids         (SubGridSet): Instance of SubGridSet
        logger       (logging.Logger): Logger
        log_level               (str): Level of the Logger

        Returns
        -------
        'NestedSampling': An instance of class NestedSampling

        Authors: Allan Denis
        '''

        logger = logger or setup_logging(level=log_level, name='NestedSampling')
        logger.debug('Building an instance of NestedSampling from dictionary')

        if not isinstance(data, dict):
            raise ForMoSAError(f'Wrong type for data: {type(data)}. Expected a dictionary', logger)

        algorithm, npoints, logL_type, config_NS, parameters, wave_fit, interp_method, bounds_lsq = data['algorithm'], data['npoints'], data['logLtype'], data['config_NS'], data['parameters'], data['wave_fit'], data['interp_method'], data['bounds_lsq']

        logger.info(f'    Found NestedSampling with {algorithm} algorithm, {npoints} living points and {len(parameters)} parameters')

        algorithm = NestedAlgorithm[algorithm.upper()]
        logL_type = [LogLikelihoodType[logL_type.upper()] for logL_type in data['logLtype']]
        config_NS = Config_NS.from_dict(config_NS)
        parameters = ParameterSet.from_dict(parameters)
        bounds_lsq = [(float(lower), float(upper)) for lower, upper in bounds_lsq]

        return cls(
            algorithm=algorithm,
            npoints=npoints,
            logL_type=logL_type,
            config_NS=config_NS,
            parameters=parameters,
            wave_fit=wave_fit,
            interp_method=interp_method,
            bounds_lsq=bounds_lsq,
            observations=observations,
            subgrids=subgrids,
            logger=logger)

    @classmethod
    def from_json(cls, path: str | os.PathLike, observations: ObservationSet, subgrids: SubGridSet, logger: logging.Logger | None=None, log_level: str='INFO') -> 'NestedSampling':
        '''
        Reconstruct an instance of NestedSampling from the folder containing the NestedSampling parameters.

        Parameters
        ----------
        path      (str | os.PathLike): Dictionary representation NestedSampling parameters
        observations (ObservationSet): Instance of ObservationSet
        subgrids         (SubGridSet): Instance of SubGridSet
        logger       (logging.Logger): Logger
        log_level               (str): Level of the Logger

        Returns
        -------
        'NestedSampling': An instance of class NestedSampling

        Authors: Allan Denis
        '''

        logger = logger or setup_logging(level=log_level, name='NestedSampling')

        logger.debug(f'Building instance of NestedSampling from the path {path}')

        # Initial checking
        if not isinstance(path, (str, os.PathLike)):
            raise ForMoSAError(f' Wrong type for path: {type(path)}. Expected a str or os.PathLike', logger)

        NS_params_file = Path(path)  / 'NS_params' / 'NS_params.json'

        if not NS_params_file.exists():
            raise ForMoSAError(f'{NS_params_file} does not exist', logger)

        with open(NS_params_file, "r") as f:
            data = json.load(f)

        ns = cls.from_dict(data, observations = observations, subgrids = subgrids, logger = logger)

        results_file = Path(path)  / 'NS_results' / f'results_{ns.algorithm}.json'

        if not results_file.exists():
            ns.logger.warning(f'{results_file} does not exist. Cannot load the results of the Nested Sampling', ns.logger)
        else:
            ns.load_results(path)

        return ns

    # =================
    # Method
    # =================

    def _validate(self) -> None:
        '''
        Validation

        Authors: Allan Denis
        '''

        for name, instance in zip(['logL_type', 'observations', 'subgrids', 'parameters', 'config_NS', 'wave_fit', 'interp_method', 'bounds_lsq'], [list, ObservationSet, SubGridSet, ParameterSet, Config_NS, list, str, list]):
            if not isinstance(getattr(self, name), instance):
                raise ForMoSAError(f'Wrong type for {name}: {type(getattr(self, name))}. Expected {instance}', self.logger)

        for name, value in zip(['logL_type', 'wave_fit', 'subgrids'], [self.logL_type, self.wave_fit, self.subgrids.subgrids]):
            if len(value) != self.observations.n_observations:
                raise ForMoSAError(f'Invalid length for {name}. Expected {self.observations.n_observations}', self.logger)

        for bound in self.bounds_lsq:
            if not isinstance(bound, tuple):
                raise ForMoSAError(f'bounds_lsq must only contain tuples. Found a {type(bound)}', self.logger)
            if len(bound) != 2:
                raise ForMoSAError(f'bounds_lsq must only contain tuples of length 2. Found a tuple of length {len(bound)}', self.logger)
            for b in bound:
                if not isinstance(b, float):
                    raise ForMoSAError(f'Wrong type detected in {bound}: {type(b)}. Expected a float', self.logger)

        valid_algorithms = [algo.value for algo in NestedAlgorithm]
        if self.algorithm not in valid_algorithms:
            raise ForMoSAError(f'{self.algorithm} is not supported. Choose from {valid_algorithms}', self.logger)

        if self.algorithm == NestedAlgorithm.NESTLE and nestle is None:
            raise ForMoSAError('nestle is not installed', self.logger)

        if self.algorithm == NestedAlgorithm.ULTRANEST and ultranest is None:
            raise ForMoSAError('ultranest is not installed', self.logger)

        if self.algorithm == NestedAlgorithm.PYMULTINEST and pymultinest is None:
            raise ForMoSAError('pymultinest is not installed', self.logger)

        valid_logL_type = [logL for logL in LogLikelihoodType]

        for i, ll in enumerate(self.logL_type):
            if ll not in valid_logL_type:
                raise ForMoSAError('Invalid logL_type ({ll}). Choose from {valid_logL_type}', self.logger)

    def _restricted_models_and_data(self) -> None:
        '''
        Create restricted versions of subgrids and observations according to wave_fit.
        These restricted instances are stored internally and reused during the nested sampling.

        Authors: Allan Denis
        '''

        self.logger.info(f'    Restrict subgris and observations to windows {self.wave_fit}')

        for index in range(self.observations.n_observations):

            # Restrict grid and observation to wave_fit
            subgrid = self.subgrids.subgrids[index]._restricted_grid(self.wave_fit[index], print_logger=False)
            observation = self.observations.observations[index]._restricted_observation(self.wave_fit[index], print_logger=False)

            # Add subgrid and observation to restricted sets
            self._restricted_subgrids.add_subgrid(subgrid)
            self._restricted_observations.add_observation(observation)

    def run(self, results_path: str | os.PathLike) -> None:
        '''
        Method to run the nested sampling algorithm using the model, observation and nested sampling parameters.

        Parameters
        ----------
        results_path   (str | os.PathLike): Path of the output

        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''

        if not isinstance(results_path, (str, os.PathLike)):
            raise ForMoSAError(f'Wrong type for results_path: {type(results_path)}. Expected a string or os.PathLike', self.logger)

        n_free_parameters = self.parameters.n_free_parameters

        loglike_gp = lambda theta: self.loglike(theta)
        prior_transform_gp = lambda theta: self.prior_transform(theta)

        time_before_ns = time.time()

        self.logger.debug('Running Nested Sampling')

        self.logger.info(f'    Algorithm for the Nested Sampling: {self.algorithm}')

        path = Path(results_path)
        self.logger.info(f'    Creating directory {path}')

        path.mkdir(exist_ok=True, parents=True)

        self.logger.info(f'    Summary for the Nested Sampling parameters: \n {self.parameters.summary(as_dataframe=True)}')

        # Algorithms
        if self.algorithm == NestedAlgorithm.NESTLE.algo:
            if nestle is None:
                msg = 'Nestle is not installed. Please install it to use the nestle algorithm.'
                self._logger.error(msg)
                raise ForMoSAError(msg)

            res = nestle.sample(loglike_gp, prior_transform_gp, n_free_parameters,
                                npoints=self.npoints,
                                **self.ns_params,
                                callback=nestle.print_progress)

            self._results = NSResults.from_nestle(res, self.parameters.free_titles)

        elif self.algorithm == NestedAlgorithm.PYMULTINEST.algo:
            if pymultinest is None:
                msg = 'Pymultinest is not installed. Please install it to use the MultiNest algorithm.'
                self._logger.error(msg)
                raise ForMoSAError(msg)

            output_path = f"{results_path}/pymultinest/"
            if not Path(output_path).exists():
                Path(output_path).mkdir(exist_ok=True, parents=True)

            res = pymultinest.solve(LogLikelihood=loglike_gp,
                                    Prior=prior_transform_gp,
                                    n_dims=n_free_parameters,
                                    n_live_points=self.npoints,
                                    outputfiles_basename=str(results_path) + '/pymultinest/' + 'RAW_',
                                    **self.ns_params)

            self._results = NSResults.from_pymultinest(output_path, self.parameters.free_titles)

        elif self.algorithm == NestedAlgorithm.ULTRANEST.algo:
            if ultranest is None:
                msg = 'Ultranest is not installed. Please install it to use the UltraNest algorithm.'
                self._logger.error(msg)
                raise ForMoSAError(msg)

            sampler = ultranest.ReactiveNestedSampler(param_names=[p.title for p in self.parameters.free_parameters],
                                    loglike=loglike_gp,
                                    transform=prior_transform_gp,
                                    log_dir=str(results_path) + '/ultranest/',
                                    **self.ns_params['ReactiveNS'])

            res = sampler.run(min_num_live_points=self.npoints, **self.ns_params['runNS'])

            self._results = NSResults.from_ultranest(res, self.parameters.free_titles)

        else:
            raise ForMoSAError(f'Unknown algorithm: {self.algorithm}. Chose amongst {[algo.value for algo in NestedAlgorithm]}')

        self._logger.info(f'    Time spent: {time.time() - time_before_ns}')

        self._logger.info(f'    Summary of Nested Sampling: \n {self.results.summary()}')

    def loglike(self, theta: np.ndarray[float]) -> float:
        '''
        Compute the loglikelihood for given parameters of the nested sampling.

        Parameters
        ----------
        theta           (np.ndarray[float]): Values randomly drawn by the Nested Sampling

        Returns
        -------
        logL (float): Final loglikelihood

        '''

        observed_models = self.build_models_from_theta(theta)
        return self._compute_loglikelihood(observed_models)

    def prior_transform(self, theta: np.ndarray[float]) -> np.ndarray[float]:
        '''
        ransform theta in [0,1]^N into physical values for free parameters.

        Parameters
        ----------
        theta (np.ndarray[float]): List of the parameters drawn by the nested sampling algorithm

        Returns
        -------
        np.ndarray[float]: Transformed values

        Authors: Allan Denis
        '''

        try:
            return self.parameters.prior_transform(theta)
        except Exception as e:
            raise ForMoSAError(e, self.logger)

    def build_models_from_theta(self, theta: np.ndarray[float]) -> list[ObservedModel]:
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

        observed_models = []
        for index in range(self.observations.n_observations):
            # Get parameters associated to the observation
            observed_params = self._build_params_for_obs(theta, index)

            # Restrict grid and observation to wave_fit
            subgrid = self.restricted_subgrids.subgrids[index]
            observation = self.restricted_observations.observations[index]

            observed_models.append(subgrid._build_model_from_params(observed_params, observation, interp_method = self.interp_method, bounds_lsq = self.bounds_lsq[index]))

        return observed_models

    def _compute_loglikelihood(self, observed_models: list[ObservedModel]) -> float:
        '''
        Compute loglikelihood.

        Parameters
        ----------
        observed_models (list[ObservedModel]): List of instances of class ObservedModel

        Returns
        -------
        float: loglikelihood value

        Authors: Allan Denis
        '''

        logL = 0
        for index in range(self.observations.n_observations):
            subgrid = self.subgrids.subgrids[index]._restricted_grid(self.wave_fit[index], print_logger=False)
            observation = self.observations.observations[index]._restricted_observation(self.wave_fit[index], print_logger=False)

            # Compute model from the parameters
            logL += subgrid._compute_loglike_for_obs(observed_models[index], observation, self.logL_type[index], interp_method = self.interp_method, bounds_lsq = self.bounds_lsq[index])

        return logL

    def _build_params_for_obs(self, free_values: np.ndarray[float], obs_index: int) -> ObservedParameters:
        '''
        Build a list of free + fixed parameters relevant to a given observation.

        Parameters
        ----------
        free_values (np.ndarray[float]): List of transformed values for the free parameters
        obs_index                 (int): Index of the observation from which we want the parameter values

        Return
        ------
        ObservedParameters: Intance of class ObservedParameters containing dictionary of parameters values (free + fixed) associated to the observation index

        Authors: Allan Denis
        '''

        if len(free_values) != self.parameters.n_free_parameters:
            raise ForMoSAError("Invalid free_values length", self.logger)

        params = {}
        free_iter = iter(free_values)

        subgrid = self.restricted_subgrids.subgrids[obs_index]

        for p in self.parameters.parameters:

            # Check if this parameter applies to the observation type
            if p.kind not in subgrid.relevant_parameter_kinds:  # We ignore this parameter if it does not apply to the observation
                continue

            # Global parameters are always included
            if not p.is_local:
                params[p] = p.prior.value if p.is_fixed else next(free_iter)

            # Local parameters are included only if obs_index matches
            elif p.is_local and obs_index in p.obs_index:
                params[p] = p.prior.value if p.is_fixed else next(free_iter)

        return ObservedParameters(params)

    def save_results(self, results_path: str | os.PathLike, **kwargs) -> None:
        '''
        Method to save the results to the path results_path

        Parameters
        ----------
        results_path (str | os.PathLike): Path to save the results to
        **kwargs                        = Additional parameters (wav_fit, interp_method, bounds_lsq)

        Authors: Allan Denis
        '''

        self._logger.info('    Saving results')

        results_file = Path(results_path)  / 'NS_results' / f'results_{self.algorithm}.json'
        NS_params_file = Path(results_path)  / 'NS_params' / 'NS_params.json'

        # NS results
        self.logger.debug(f'Saving results to path {results_file}>')
        results_file.parent.mkdir(exist_ok=True, parents=True)
        with open(results_file, 'w') as f:
            json.dump(self.results.to_dict, f, indent=4)

        # NS params
        self.logger.debug(f'Saving NS paramrs to path {NS_params_file}')
        NS_params_file.parent.mkdir(exist_ok=True, parents=True)
        with open(str(NS_params_file), 'w') as f:
            json.dump(self.to_dict, f, indent=4)

    def load_results(self, results_path: str | os.PathLike) -> None:
        '''
        Method to load the results from the path results_paths

        Parameters
        ----------
        results_path (str | os.PathLike): Path to save the results to

        Authors: Allan Denis
        '''

        results_path = Path(results_path)  / 'NS_results'

        self.logger.debug(f'Loading results from path {results_path}')

        results_file = results_path / f'results_{self.algorithm}.json'

        if not(results_file.exists()):
            raise ForMoSAError(f'<{results_file} does not exist. Please make sure to use an existing result file>', self.logger)

        self._logger.debug(f'< load {results_file}')
        with open(results_file, 'r') as f:
            results = json.load(f)

        self._results = NSResults.from_dict(results)

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
        if self.results is None:
            raise ForMoSAError('Please first run the Nested Sampling before computing the best fit', self.logger)

        if perc < 0 or perc > 1:
            raise ForMoSAError(f'perc must be a float between 0 and 1. Got {perc} with type {type(perc)}', self.logger)

        lower = (1 - perc) / 2
        upper = (1 + perc) / 2

        wrange = (self.restricted_observations.wavelength_range[0]*0.95, self.restricted_observations.wavelength_range[1]*1.05)
        grid = self.restricted_subgrids.parent_grid._restricted_grid(f'{wrange[0]},{wrange[1]}', print_logger=True)

        models_flux = []

        self.logger.info(f'    Computing confidence interval with percentiles [{np.round(lower,2)} - {np.round(upper,2)}]')
        for sample in tqdm(self.results.samples):
            observed_params = self._build_params_for_obs(sample, 0).global_params

            observed_model = ObservedModel.from_grid_and_params(grid, observed_params)

            models_flux.append(observed_model.flux)

        models_flux = np.array(models_flux)

        perc_1sigma_lower = get_weighted_percentile(lower, models_flux, weights=self.results.weights)
        perc_1sigma_higher = get_weighted_percentile(upper, models_flux, weights=self.results.weights)

        return ObservedModel(observed_model.wave, perc_1sigma_lower, observed_model.res), ObservedModel(observed_model.wave, perc_1sigma_higher, observed_model.res)
