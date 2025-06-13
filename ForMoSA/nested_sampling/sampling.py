import numpy as np
import ForMoSA.utils.spec as us
import ForMoSA.utils.hc as hc
import ForMoSA.utils.logL_functions as logL_functions
import os
import time
import pickle
import astropy.constants as cst

from pathlib import Path
from scipy.interpolate import interp1d

from ForMoSA.model_grid import ModelGrid, ModelSubGrid
from ForMoSA.observation import Observation
from .plotting import NestedSamplingPlotting
from .parameters import NestedSamplingParameters, Parameter
from ForMoSA.error import ForMoSAError

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
    from ultranest import integrator
except ImportError:
    ultranest = None


class NestedSampling(object):
    '''
    ForMoSA Nested_Sampling class, which provides easy access to the parameters of the nested sampling algorithm

    Parameters
    ----------

    logger          (Logger): Logger
    algorithm          (str): Algorithm used for the nested sampling ('nestle', 'ultranest' or 'pymultinest')
    npoints            (int): Number of living points used for the nested sampling
    logL_type         (list):
    config_ns_algo    (dict): Dictionary containing the parametes of the different nested sampling algorithm

    Authors: Allan Denis
    '''

    def __init__(self, algorithm: str, npoints: int, logL_type: list, logger, config_ns_algo: dict):
        self._logger = logger
        self._logL = dict()

        valid_algorithms = ['nestle', 'ultranest', 'pymultinest']
        if algorithm not in valid_algorithms:
            msg = f" {algorithm} is not a supported algorithm. Please choose amongst {', '.join(valid_algorithms)}"
            self._logger.error(msg)
            raise ForMoSAError(msg)

        valid_logL_type = ['chi2', 'chi2_covariance', 'CCF_Brogi', 'CCF_Zucker', 'CCF_custom', 'chi2_noisescaling', 'chi2_noisescaling_covariance']
        for indobs in range(len(logL_type)):
            if logL_type[indobs] not in valid_logL_type:
                msg = f' Invalid loglikelihood type. Please choose amongst {valid_logL_type}'
                self._logger.critical(msg)
                raise ForMoSAError(msg)
            else:
                self._logL[indobs] = logL_type[indobs]

        self._algorithm = algorithm
        self._npoints = npoints
        self._logger = logger
        self._params = NestedSamplingParameters(logger)
        self._plotting = None
        self._results = None
        self._modif_data = dict()
        self._best_model = dict()

        try:   # Check if the dictionary contains the keys 'nestle', 'ultranest' and 'pymultinest'
            if algorithm == 'nestle':
                self._ns_params = config_ns_algo['nestle']
            elif algorithm == 'ultranest':
                self._ns_params = config_ns_algo['ultranest']
            elif algorithm == 'pymultinest':
                self._ns_params = config_ns_algo['pymultinest']
        except KeyError as e:
            self._logger.error(e)
            raise ForMoSAError(e)


    ##################################################
    # Representation
    ##################################################

    def __repr__(self):
        return f'<Nested sampling, algorithm = {self.algorithm}, npoints = {self.npoints}>'

    def __format__(self) -> str:
        return self.__repr__()


    ##################################################
    # Properties
    ##################################################


    @property
    def algorithm(self) -> str:                       # Algorithm
        return self._algorithm

    @property
    def logL(self) -> dict:                           # logL function
        return self._logL

    @property
    def n_obs(self) -> int:                           # Number of observations
        return self.grid.n_obs

    @property
    def params(self) -> NestedSamplingParameters:     # Priors parameters
        return self._params

    @property
    def ns_params(self) -> dict:                      # Nested sampling parameters
        return self._ns_params

    @property
    def npoints(self) -> int:                         # Nested sampling number of living points
        return self._npoints

    @property
    def results(self) -> dict:                        # Results
        return self._results

    @property
    def param_samples_dict(self) -> dict:             # Samples of each parameter
        if not hasattr(self, "results") or self.results is None:
            msg = 'No results found. Please run the sampling algorithm first.'
            self._logger.error(msg)
            raise ForMoSAError(msg)

        return {
            name: self.results['samples'][:, i]
            for i, name in enumerate(self.params.list_free_params_names)
        }

    @property
    def param_best_dict(self) -> dict:               # Best value of each parameter
        if not hasattr(self, "results") or self.results is None:
            msg = 'No results found. Please run the sampling algorithm first.'
            self._logger.error(msg)
            raise ForMoSAError(msg)

        return {
            name: np.average(self.results['samples'][:, i], weights=self.results['weights'])
            for i, name in enumerate(self.params.list_free_params_names)
        }

    @property
    def list_best_params(self) -> list:              # List of best values for each parameter
        return list(self.param_best_dict.values())

    @property
    def best_logL(self) -> float:                    # Averaged value of logL
        if not hasattr(self, "results") or self.results is None:
            msg = 'No results found. Please run the sampling algorithm first.'
            self._logger.error(msg)
            raise ForMoSAError(msg)
        return np.average(self.results['logl'], weights = self.results['weights'])

    @property
    def modif_data(self) -> dict:                    # Modified data
        if not hasattr(self, "results") or self.results is None:
            msg = 'No results found. Please run the sampling algorithm first.'
            self._logger.error(msg)
            raise ForMoSAError(msg)
        return self._modif_data

    @property
    def best_model(self) -> dict:                    # Best model
        if not hasattr(self, "results") or self.results is None:
            msg = 'No results found. Please run the sampling algorithm first.'
            self._logger.error(msg)
            raise ForMoSAError(msg)
        return self._best_model

    @property
    def plotting(self) -> NestedSamplingPlotting:    # NestedSamplingPlotting class
        return self._plotting


    def run(self, results_path: str | os.PathLike, observation: Observation, modelgrid: ModelGrid, interp_method: str = 'linear', wav_cont: list = ['NA'], res_cont: list = ['NA'], bounds_lsq: list = [('NA', 'NA')], emulator: list = ['NA']) -> None:
        '''
        Method to run the nested sampling algorithm using the model, observation and nested sampling parameters.

        Parameters
        ----------
        logL_type                  (list): Loglikelihood function  (['chi2'], ['chi2_covariance'], ['CCF_Brogi'], ['CCF_Zucker'], ...)
        results_path   (str | os.PathLike): Path of the output
        observation         (Observation): Inbstance of :class:`~ForMoSA.Observation`
        modelGrid           (Observation): Inbstance of :class:`~ForMoSA.ModelGrid`
        interp_method               (str): Interpolation method ('linear', 'cubic', 'spline', ...)
        wav_cont                   (list): List of wavelength grid used for the continuum (used for high contrast)
        res_cont                   (list): List of resolution used for the continuum (used for high contrast)
        bounds_lsq                 (list): List of bounds used for the least squares (used for high contrast)
        emulator                   (list): Emulator of the grid ('PCA', 'NMF')

        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''

        self._logger.info(f' Run Nested Sampling algorithm using {self.npoints} living points and {self.algorithm}.')

        # Replace 'parX' names by associated physical parameters ('Teff', 'logg', ...)
        for name in self.params.parameters.keys():
            if name.startswith('par'):  # Detect grid parameters
                self.params.parameters[name]._name = modelgrid.titles[modelgrid.keys.index(name)]  # Rename parameter with title associated to 'parX'


        res_mod_obs_list = []
        for indobs in range(observation.n_obs):
            if (bounds_lsq[indobs % len(bounds_lsq)] == ('NA', 'NA')) and (not(self.logL[indobs % len(self.logL)].startswith('CCF'))) and (len(observation.obs_data[indobs]['spectro']['star_flx']) > 0):
                msg = f' If you do not chose a CCF mapping loglikelihood function ({self.logL[indobs]}), please chose values for the LSQ bounds'
                self._logger.error(msg)
                raise ForMoSAError(msg)

            obs_data_spectro, mod_data_spectro = observation.obs_data[indobs]['spectro'], modelgrid.adapted_grid[indobs]['spectro']

            if len(obs_data_spectro['inv_cov']) > 0 and not(self.logL[indobs % len(self.logL)].endswith('_covariance')):
                self._logger.warning(f' observation {observation.obs_name[indobs]} contains a covariance matrix but your loglikelihood function does not account for covariance matrices. You should consider changing the loglikelihood function from {self.logL[indobs]} to {self.logL[indobs] + "_covariance"}.')

            if len(obs_data_spectro['inv_cov']) == 0 and self.logL[indobs % len(self.logL)].endswith('_covariance'):
                msg = f' You chose a loglikelihood accounting for covariance matrices but your observation {observation.obs_name[indobs]} does not contain any covariance matrix. Please adapt your loglikelihood function.'
                self._logger.error(msg)
                raise ForMoSAError(msg)

            if len(obs_data_spectro['wav']) > 0:
                res_mod_obs = obs_data_spectro['res']
                res, target_wavelength = mod_data_spectro.resolution, obs_data_spectro['wav']
                if (len(res) != len(target_wavelength)):
                    interp_mod_to_obs = interp1d(mod_data_spectro.wavelength, mod_data_spectro.resolution, fill_value='extrapolate')
                    res_mod_obs = interp_mod_to_obs(obs_data_spectro['wav'])

            else:
                res_mod_obs = 0

            res_mod_obs_list.append(res_mod_obs)

        n_free_parameters = self._params.n_free_parameters

        loglike_gp = lambda theta: self._loglike(theta, observation, modelgrid, res_mod_obs_list, interp_method=interp_method, wav_cont=wav_cont, res_cont=res_cont, bounds_lsq=bounds_lsq, emulator=emulator)
        prior_transform_gp = lambda theta: self._prior_transform(theta, modelgrid)

        os.makedirs(str(results_path) + f'/{self.algorithm}/', exist_ok=True)

        time1 = time.time()

        if self.algorithm == 'nestle':
            if nestle is None:
                msg = 'Nestle is not installed. Please install it to use the nestle algorithm.'
                self._logger.error(msg)
                raise ForMoSAError(msg)

            res = nestle.sample(loglike_gp, prior_transform_gp, n_free_parameters,
                                npoints=self.npoints,
                                **self.ns_params,
                                callback=nestle.print_progress)


            logz = [res['logz'], res['logzerr']]
            samples = res['samples']
            weights = res['weights']
            logvol = res['logvol']
            logl = res['logl']

        elif self.algorithm == 'pymultinest':
            if pymultinest is None:
                msg = 'Pymultinest is not installed. Please install it to use the MultiNest algorithm.'
                self._logger.error(msg)
                raise ForMoSAError(msg)

            res = pymultinest.solve(LogLikelihood=loglike_gp,
                                    Prior=prior_transform_gp,
                                    n_dims=n_free_parameters,
                                    n_live_points=self.npoints,
                                    outputfiles_basename=str(results_path) + '/pymultinest/' + 'RAW_',
                                    **self.ns_params)

            # Reformat the result file
            with open(str(results_path) + '/pymultinest/' + 'RAW_stats.dat', 'rb') as f:
                line = f.readline().strip().split()
                logz = [float(line[5]), float(line[7])]

            sample_multi, logl_multi, logvol_multi = [], [], []
            with open(str(results_path) + '/pymultinest/' + 'RAW_ev.dat', 'rb') as f:
                for line in f:
                    parts = line.strip().split()
                    sample_multi.append([float(p) for p in parts[:-3]])
                    logl_multi.append(float(parts[-3]))
                    logvol_multi.append(float(parts[-2]))

            samples, weights, logl, logvol = [], [], [], []
            with open(str(results_path) + '/pymultinest/' + 'RAW_.txt', 'rb') as f:
                for line in f:
                    parts = line.strip().split()
                    point = [float(p) for p in parts[2:]]
                    if point in sample_multi:
                        idx = sample_multi.index(point)
                        samples.append(point)
                        weights.append(float(parts[0]))
                        logl.append(logl_multi[idx])
                        logvol.append(logvol_multi[idx])

            samples = np.asarray(samples)
            weights = np.asarray(weights)
            logl = np.asarray(logl)
            logvol = np.asarray(logvol)
            logz = np.asarray(logz)

        elif self.algorithm == 'ultranest':
            if ultranest is None:
                msg = 'Ultranest is not installed. Please install it to use the UltraNest algorithm.'
                self._logger.error(msg)
                raise ForMoSAError(msg)

            sampler = ultranest.ReactiveNestedSampler(param_names=self.params.list_free_params_keys,
                                    loglike=loglike_gp,
                                    transform=prior_transform_gp,
                                    log_dir=str(results_path) + '/ultranest/',
                                    **self.ns_params)

            sampler.run(min_num_live_points=self.npoints, **self.ns_params)

            res = integrator.read_file(str(results_path) + '/ultranest/',
                                    x_dim=self.params.n_free_parameters,
                                    num_bootstraps=self.ns_params['num_bootstraps'])

            logz = [res[-1]['logz'], res[-1]['logzerr']]
            samples = res[-1]['samples']
            weights = res[-1]['weighted_samples']['weights']
            logvol = res[0]['logvol']  # Not always used in UltraNest
            logl = res[0]['logl']


        self._results = {"samples": samples,
                         "weights": weights,
                         "logl": logl,
                         "logvol": logvol,
                         "logz": logz}

        time_elapsed = time.time() - time1
        if time_elapsed < 60:
            time_spent = f'{time_elapsed:.1f} sec'
        elif time_elapsed < 3600:
            time_spent = f'{time_elapsed/60:.1f} min'
        else:
            time_spent = f'{time_elapsed/3600:.1f} hours'

        self._fitted = True

        self._logger.info(f' Time spent: {time_spent}')

        self._logger.info(f'Summary of Nested Sampling : \n {self._summary()}')


    def _loglike(self, theta: list, observation: Observation, modelgrid: ModelGrid, res_mod_obs: list, wav_cont: list = ['NA'], res_cont: list = ['NA'], bounds_lsq: list = ['NA'], interp_method: str = 'linear', emulator: list = ['NA']) -> float | tuple[dict, np.ndarray, np.ndarray]:
        '''
        Compute the loglikelihood for given values of the parameters of the nested sampling

        Parameters
        ----------
        theta                      (list): Parameters values picked by the nested sampling
        observation         (Observation): Inbstance of :class:`~ForMoSA.Observation`
        modelGrid           (Observation): Inbstance of :class:`~ForMoSA.ModelGrid`
        res_mod_obs                (list): Resolution of the spectroscopic of each adapted model interpolated onto the wavelength grid of the corresponding observation
        wav_cont                   (list): List of wavelength grid used for the continuum (used for high contrast)
        res_cont                   (list): List of resolution used for the continuum (used for high contrast)
        bounds_lsq                 (list): List of bounds used for the least squares (used for high contrast)
        emulator                   (list): Emulator of the grid ('PCA', 'NMF')

        Returns:
            - FINAL_logL     (float): Final evaluated loglikelihood for both spectra and photometry.

        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''

        if emulator[0] == 'PCA':
            # TODO
            pass

        try:
            FINAL_logL = 0
            modif_data, modif_model = dict(), dict()
            for indobs in range(observation.n_obs):
                # Modified spectro
                modif_data[indobs], modif_model[indobs] = self._compute_model_from_theta(theta, observation.obs_data[indobs]['spectro'], observation.obs_data[indobs]['photo'], modelgrid.adapted_grid[indobs]['spectro'], modelgrid.adapted_grid[indobs]['photo'], res_mod_obs[indobs % len(res_mod_obs)], interp_method = interp_method, wav_cont = wav_cont[indobs % len(wav_cont)], res_cont = res_cont[indobs % len(res_cont)], bounds_lsq = bounds_lsq[indobs % len(bounds_lsq)], indobs = indobs)
                logL = self._compute_loglike_from_model_and_spectra(modif_data[indobs]['spectro'], modif_data[indobs]['photo'], modif_model[indobs]['spectro'], modif_model[indobs]['photo'], indobs = indobs)

                # Increment total Log-likelihood
                FINAL_logL += logL

        except ForMoSAError as e:
            self._logger.error(f"Error computing loglikelihood: {e}")
            raise ForMoSAError(e)

        return FINAL_logL


    def _compute_model_from_theta(self, theta: list, obs_dict_spectro: dict, obs_dict_photo, grid_spectro: ModelGrid | ModelSubGrid, grid_photo: ModelGrid | ModelSubGrid, res_mod_obs_spectro: np.ndarray, interp_method: str = 'linear', wav_cont: str | np.ndarray = 'NA', res_cont: str | np.ndarray = 'NA', bounds_lsq: list = ['NA', 'NA'], indobs: int = 0) -> tuple[dict, dict]:
        '''
        Method to modify the interpolated synthetic spectra with the different extra-grid parameters.
        It can perform : Re-calibration on the data, Doppler shifting, Application of a substellar extinction, Application of a rotational velocity,
        Application of a circumplanetary disk (CPD).

        Parameters
        ----------
        theta                                (list): Parameter values randomly picked by the nested sampling
        obs_dict_spectro                     (dict): Dictionary of observation spectroscopic data {'wav': wav, 'flx': flx, ...}
        obs_dict_photo                       (dict): Dictionary of observation photometric data {'wav': wav, 'flx': flx, ...}
        grid_spectro     (ModelGrid | ModelSubGrid): Instance of :class:'~ModelGrid' or :class:'~ModelSubGrid' adapted to spectroscopic data
        grid_photo       (ModelGrid | ModelSubGrid): Instance of :class:'~ModelGrid' or :class:'~ModelSubGrid' adapted to photometric data
        res_mod_obs_spectro            (np.ndarray): Resolution of the adapted model interpolated onto the wavelength grid of the observation
        interp_method                         (str): Method for the interpolation of the grid
        wav_cont                 (str | np.ndarray): Wavelength grid for the continuum estimation of the model (used for high contrast)
        res_cont                 (str | np.ndarray): Resolution of the continuum (used for high contrast)
        bounds_lsq                           (list): Bounds of the least squares estumatiion (used for high contrast)
        indobs                                (int): Index of the current observation looping

        Returns:
            - obs_dict   (dict): Dictionary of modified spectra {'spectro': dict, 'photo': dict}
            - mod_dict   (dict): Dictionary of modified model {'spectro': dict, 'photo: dict'}

        Author: Simon Petrus, Paulina Palma-Bifani, Allan Denis and Matthieu Ravet
        '''
        def get_param(name, indobs):
            name = name if name in self.params.parameters else f"{name}_{indobs}" if f"{name}_{indobs}" in self.params.parameters else None   # treat also the multi-observations parameters
            if name is None:
                return None
            new_theta = self.params._get_param_value(name, theta)
            return new_theta

        contributions, obs_dict_spectro['speckles'], ck_spectro, ck_photo = 1, np.zeros(len(obs_dict_spectro['wav'])), 1, 1

        theta_index = self.params.list_params_keys
        if len(theta) != self._params.n_free_parameters:
            msg = f"theta length ({len(theta)}) does not match expected number of free parameters ({self._params.n_free_parameters})"
            self._logger.critical(msg)
            raise ForMoSAError(msg)

        theta_grid = [theta[i] if key in self.params.list_free_params_keys
                      else self.params.parameters[key].value
                      for i, key in enumerate(theta_index)
                      if key.startswith('par')]

        if res_cont != 'NA':
            res_cont = float(res_cont)

        # Retrieve usefull parameters
        wav_mod_spectro = grid_spectro.wavelength
        wav_mod_photo = grid_photo.wavelength
        target_wavelength, target_resolution = obs_dict_spectro['wav'], obs_dict_spectro['res']
        ins_spectro, ins_photo = grid_spectro.instrument, grid_photo.instrument

        # Interpolate at the values of the grid parameters
        flx_mod_spectro, flx_mod_photo = grid_spectro._interpolate_between_gridpoints(theta_grid, interp_method, indobs), grid_photo._interpolate_between_gridpoints(theta_grid, interp_method, indobs)

        # RV correction
        rv = get_param('rv', indobs)
        if rv is not None:
            wav_mod_spectro, flx_mod_spectro = us.doppler_fct(wav_mod_spectro, flx_mod_spectro, rv)

        # vsini correction
        vsini = get_param('vsini', indobs)
        ld = get_param('ld', indobs)
        if vsini is not None and ld is not None:
            try:
                vsini_function = str(self.params.parameters['vsini'].vsini_function)
            except KeyError:
                vsini_function = str(self.params.parameters[f'vsini_{indobs}'].vsini_function)
            flx_mod_spectro, res_mod_obs_spectro = us.vsini_fct(wav_mod_spectro, flx_mod_spectro, res_mod_obs_spectro, ld, vsini, vsini_function)

        # Reddening
        av = get_param('av', indobs)
        if av is not None:
            flx_mod_spectro, flx_mod_photo = us.reddening_fct(wav_mod_spectro, obs_dict_photo['wav'], flx_mod_spectro, flx_mod_photo, av)

        # CPD
        bb_T = get_param('bb_T', indobs)
        bb_R = get_param('bb_R', indobs)
        d = get_param('d', indobs)
        if None not in (bb_T, bb_R, d):
            flx_mod_spectro, flx_mod_photo = us.bb_cpd_fct(wav_mod_spectro, obs_dict_photo['wav'], flx_mod_spectro, flx_mod_photo, d, bb_T, bb_R)

        # Save native model before resampling
        flx_mod_spectro_nativ = np.copy(flx_mod_spectro)
        # Resolution decreasing and resampling
        if len(wav_mod_spectro) != len(obs_dict_spectro['wav']):
            flx_mod_spectro = us.resolution_decreasing(wav_mod_spectro, flx_mod_spectro, res_mod_obs_spectro, target_wavelength, target_resolution)

        # High contrast modeling
        if len(obs_dict_spectro['star_flx']) > 0:
            flx_cont_mod_spectro = us.continuum_estimate(target_wavelength, flx_mod_spectro, res_mod_obs_spectro, wav_cont, res_cont)
            if self.logL[indobs % len(self.logL)].startswith('chi2'):
                contributions, flx_mod_spectro, obs_dict_spectro['speckles'] = hc._hc_model_estimate_speckles(obs_dict_spectro['flx'], obs_dict_spectro['flx_cont'], obs_dict_spectro['transm'], obs_dict_spectro['star_flx'], obs_dict_spectro['star_flx_cont'], flx_mod_spectro, flx_cont_mod_spectro, obs_dict_spectro['err'], bounds_lsq, obs_dict_spectro['system'])
            else:
                flx_mod_spectro, obs_dict_spectro['speckles'] = hc._hc_model_remove_speckles(obs_dict_spectro['flx'], obs_dict_spectro['flx_cont'], obs_dict_spectro['transm'], obs_dict_spectro['star_flx'], obs_dict_spectro['star_flx_cont'], flx_mod_spectro, flx_cont_mod_spectro)

        # Scaling (ck)
        alpha = get_param('alpha', indobs)
        if alpha is None:
            alpha = 1
        r = get_param('r', indobs)
        if len(obs_dict_spectro['star_flx']) == 0 and r is not None and d is not None:
            flx_mod_spectro, flx_mod_photo, ck_spectro, ck_photo = us.calc_ck(obs_dict_spectro, obs_dict_photo, flx_mod_spectro, flx_mod_photo, r, d, alpha or 0)
        # Analytical resolution and special case for MOSAIC when you don't fit for R and D for one of the obs but still want to fit it for the others
        elif len(obs_dict_spectro['star_flx']) == 0:
            flx_mod_spectro, flx_mod_photo, ck_spectro, ck_photo = us.calc_ck(obs_dict_spectro, obs_dict_photo, flx_mod_spectro, flx_mod_photo, 0, 0, alpha=0, analytic='yes')

        mod_dict_spectro = {'wav': wav_mod_spectro, 'flx': flx_mod_spectro, 'nativ_flx': flx_mod_spectro_nativ, 'res': obs_dict_spectro['res'], 'ins': ins_spectro, 'ck': ck_spectro, 'hc_contributions': contributions}
        mod_dict_photo = {'wav': wav_mod_photo, 'flx': flx_mod_photo, 'ins': ins_photo, 'ck': ck_photo}

        obs_dict = {'spectro': obs_dict_spectro, 'photo': obs_dict_photo}
        mod_dict = {'spectro': mod_dict_spectro, 'photo': mod_dict_photo}

        return obs_dict, mod_dict


    def _compute_loglike_from_model_and_spectra(self, obs_dict_spectro: dict, obs_dict_photo: dict, mod_dict_spectro: dict, mod_dict_photo: dict, indobs: int = 0):
        '''
        Method to compute the loglikelihood from the modified observation and model

        Parameters
        ----------
        obs_dict_spectro    (dict): Dictionary of observation spectroscopic data modified by the nested sampling {'wav': wav, 'flx': flx, ...}
        obs_dict_photo      (dict): Dictionary of observation photometric data modified by the nested sampling {'wav': wav, 'flx': flx, ...}
        mod_dict_spectro    (dict): Dictionary of model spectroscopic data modified by the nested sampling {indobs: {'spectro': dict 'photo': dict}}
        mod_dict_photo      (dict): Dictionary of model photometric data modified by the nested sampling {indobs: {'spectro': dict 'photo': dict}}
        indobs               (int): Index of current observation

        Returns:
            - Final_logL (float): Final loglikelihood value

        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''

        # LogL Photometry
        logL_photo = 0 if len(obs_dict_photo['wav']) == 0 else logL_functions.logL_chi2(obs_dict_photo['flx'] - mod_dict_photo['flx'], obs_dict_photo['err'])

        # LogL Spectroscopy
        logL_spectro = 0
        if len(obs_dict_spectro['wav']) > 0:
            residual = obs_dict_spectro['flx'] - mod_dict_spectro['flx'] - obs_dict_spectro['speckles']
            ll_type = self.logL[indobs % len(self.logL)]
            logL_dict = {'chi2': lambda: logL_functions.logL_chi2(residual, obs_dict_spectro['err']),
                         'chi2_covariance': lambda: logL_functions.logL_chi2_covariance(residual, obs_dict_spectro['inv_cov']),
                         'CCF_Brogi': lambda: logL_functions.logL_CCF_Brogi(obs_dict_spectro['flx'] - obs_dict_spectro['speckles'], mod_dict_spectro['flx']),
                         'CCF_Zucker': lambda: logL_functions.logL_CCF_Zucker(obs_dict_spectro['flx'] - obs_dict_spectro['speckles'], mod_dict_spectro['flx']),
                         'CCF_custom': lambda: logL_functions.logL_CCF_custom(obs_dict_spectro['flx'] - obs_dict_spectro['speckles'], mod_dict_spectro['flx'], obs_dict_spectro['err']),
                         'chi2_noisescaling': lambda: logL_functions.logL_chi2_noisescaling(residual, obs_dict_spectro['err']),
                         'chi2_noisescaling_covariance': lambda: logL_functions.logL_chi2_noisescaling_covariance(residual, obs_dict_spectro['inv_cov'])}
            logL_spectro = logL_dict.get(ll_type, lambda: 0)()

        FINAL_logL = logL_photo + logL_spectro

        if FINAL_logL < -1e6:
            self._logger.warning(f"[loglike WARNING] Unusually low loglike: {FINAL_logL}")
            for name in self.params.parameters:
                value = self.params._get_param_value(name, self.params.theta)
                self._logger.warning(f"[low loglike] {name} = {value}")

            self._logger.info(f"LogL_spectro for obs {indobs}: {logL_spectro}")
            self._logger.info(f"LogL_photo for obs {indobs}: {logL_photo}")
            self._logger.info(f"Total LogL after obs {indobs}: {FINAL_logL}")

        return FINAL_logL


    def _prior_transform(self, theta: list, modelGrid: ModelGrid) -> list:
        '''
        Method to define the priors to be used for the inversion.
        We check that the boundaries are consistent with the grid extension.

        Parameters
        ----------
        theta               (list): Parameter values randomly picked by the nested sampling
        modelGrid      (ModelGrid): Instance of :class:'~ModelGrid'

        Return:
            - prior   (list): List of parameter values transformed by the prior laws, in the original order

        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''

        prior = []
        theta_index_free = self.params.free_parameters.keys()   # List of free (without constant priors) parameters

        for i, param_name in enumerate(theta_index_free):
            param = self.params.parameters[param_name]
            theta_val = theta[i]
            prior_value = param._apply_prior(theta_val)

            if param_name.startswith('par'):
                # Clamp within the grid bounds
                prior_value = max(min(prior_value, modelGrid.lims_params_grid[param_name][1]), modelGrid.lims_params_grid[param_name][0])
                param._theta = prior_value

            prior.append(prior_value)

        self.params._theta = prior    # Update the current drawn values for the parameters

        return prior


    def _save_results(self, results_path: str | os.PathLike) -> None:
        '''
        Method to save the results to the path results_path

        Parameters
        ----------
        results_path (str | os.PathLike): Path t o save the results to

        Authors: Allan Denis
        '''

        self._logger.info(' Save results')

        results_file = Path(results_path)  / f'results_{self.algorithm}.pic'

        self._logger.debug(f'< Save to path {results_file}')
        with open(results_file, 'wb') as f:
            pickle.dump(self._results, f)


    def _load_results(self, results_path: str | os.PathLike) -> None:
        '''
        Method to load the results from the path results_paths

        Parameters
        ----------
        results_path (str | os.PathLike): Path to save the results to

        Authors: Allan Denis
        '''

        self._logger.info(' Load results')

        results_file = Path(results_path)  / f'results_{self.algorithm}.pic'

        if not(results_file.exists()):
            self._logger.error(f' {results_file} does not exist. Please make sure to use an existing result file.')

        self._logger.debug(f'< load {results_file}')
        with open(results_file, 'rb') as f:
            results = pickle.load(f)

        logz, samples, weights, logvol, logl = [results['logz'][0], results['logz'][1]], results['samples'], results['weights'], results['logvol'], results['logl']
        self._results = {"samples": samples,
                         "weights": weights,
                         "logl": logl,
                         "logvol": logvol,
                         "logz": logz}

        # Luminosity derivation
        if 'r' in self.params.list_free_params_names and 'Teff' in self.params.list_free_params_names:
            r_samples = self.param_samples_dict['r']
            Teff_samples = self.param_samples_dict['Teff']

            # Stefan-Boltzmann law
            lum = np.log10(4 * np.pi * (r_samples * cst.R_jup.value) ** 2 * Teff_samples ** 4 * cst.sigma_sb.value / cst.L_sun.value)
            lum_param = Parameter(r'log(L/L$\mathrm{_{\odot}}$)', 'computed')

            self._results['samples'] = np.hstack([self._results['samples'], lum[:, None]])
            self.params._add_parameter(lum_param, r'log(L/L$\mathrm{_{\odot}}$)')


    def _compute_best_model(self, observation: Observation, modelgrid: ModelGrid, interp_method: str = 'linear', wav_cont: list = ['NA'], res_cont: list = ['NA'], bounds_lsq: list = ['NA', 'NA']) -> None:
        '''
        Method to compute best model from nested sampling output

        Parameters
        ----------
        theta                      (list): Parameters values picked by the nested sampling
        observation         (Observation): Instance of :class:`~ForMoSA.Observation`
        modelgrid           (Observation): Instance of :class:`~ForMoSA.ModelGrid`
        wav_cont                   (list): List of wavelength grid used for the continuum (used for high contrast)
        res_cont                   (list): List of resolution used for the continuum (used for high contrast)
        bounds_lsq                 (list): List of bounds used for the least squares (used for high contrast)
        emulator                   (list): Emulator of the grid ('PCA', 'NMF')

        Authors: Allan Denis
        '''

        best_theta = self.list_best_params

        modif_data, best_model = dict(), dict()

        for indobs in range(observation.n_obs):
            obs_data_spectro, mod_data_spectro = observation.obs_data[indobs]['spectro'], modelgrid.adapted_grid[indobs]['spectro']
            if len(obs_data_spectro['wav']) > 0:
                res_mod_obs = obs_data_spectro['res']
                res, target_wavelength = mod_data_spectro.resolution, obs_data_spectro['wav']
                if (len(res) != len(target_wavelength)):
                    interp_mod_to_obs = interp1d(mod_data_spectro.wavelength, mod_data_spectro.resolution, fill_value='extrapolate')
                    res_mod_obs = interp_mod_to_obs(obs_data_spectro['wav'])

            else:
                res_mod_obs = 0

            modif_data[indobs], best_model[indobs] = self._compute_model_from_theta(best_theta, observation.obs_data[indobs]['spectro'], observation.obs_data[indobs]['photo'], modelgrid.adapted_grid[indobs]['spectro'], modelgrid.adapted_grid[indobs]['photo'], res_mod_obs, interp_method = interp_method, wav_cont = wav_cont[indobs % len(wav_cont)], res_cont = res_cont[indobs % len(res_cont)], bounds_lsq = bounds_lsq[indobs % len(bounds_lsq)], indobs = indobs)

        self._modif_data = modif_data
        self._best_model = best_model


    def _summary(self, sigma: int = 2) -> None:
        '''
        Method to print a summary of the nested sampling results including weighted statistics.

        Parameters
        ----------
        sigma     (int): Confidence interval (1 or 2 sigma), default is 1.

        Authors: Allan Denis
        '''

        if not hasattr(self, '_results') or self._results is None:
            msg = 'No results found. Please run the sampling algorithm first.'
            self._logger.error(msg)
            raise ForMoSAError(msg)

        logz, logzerr = self._results['logz']
        samples = np.asarray(self._results['samples'])
        weights = np.asarray(self._results['weights'])

        msg = "\n======== Nested Sampling Summary ========\n"
        msg += f"Algorithm            : {self.algorithm}\n"
        msg += f"LogZ                 : {logz:.3f} ± {logzerr:.3f}\n"
        msg += f"Number of samples    : {len(samples)}\n"
        msg += f"Number of parameters : {samples.shape[1] if samples.ndim > 1 else 1}\n"

        # Normalize weights
        if len(weights) != len(samples):
            print("\nWarning: Weights and samples have inconsistent sizes.")
            return
        weights /= np.sum(weights)

        # Confidence interval percentiles
        if sigma == 1:
            low_pct, high_pct = 16, 84
        elif sigma == 2:
            low_pct, high_pct = 2, 98
        else:
            raise ValueError("sigma must be 1 or 2.")

        msg += "\nPosterior (weighted):\n"
        for i in range(samples.shape[1]):
            param_samples = samples[:, i]

            # Weighted mean
            mean = np.average(param_samples, weights=weights)

            # Weighted percentiles
            sorted_indices = np.argsort(param_samples)
            sorted_samples = param_samples[sorted_indices]
            sorted_weights = weights[sorted_indices]
            cumsum_weights = np.cumsum(sorted_weights)

            def weighted_percentile(p):
                return np.interp(p / 100, cumsum_weights, sorted_samples)

            low = weighted_percentile(low_pct)
            high = weighted_percentile(high_pct)

            plus  = high - mean
            minus = low - mean
            msg += f" {self.params.list_free_params_names[i]:10s}: {mean:10.4f} {minus:+10.4f} {plus:+10.4f} [{low:10.4f}, {high:10.4f}] ({sigma}σ)\n"

        msg += "=========================================\n"

        return msg




