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
        self._burn_in = 0
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
    def burn_in(self):                 # Burn-in to apply to the chains
        return self._burn_in

    @burn_in.setter                    # Burn-in setter
    def burn_in(self, burn_in):
        self._burn_in = burn_in
        return burn_in

    @property
    def param_samples_dict(self) -> dict:             # Samples of each parameter
        if not hasattr(self, "results") or self.results is None:
            msg = 'No results found. Please run the sampling algorithm first.'
            self._logger.error(msg)
            raise ForMoSAError(msg)


        return {
            name: self.results['samples'][self.burn_in:, i]
            for i, name in enumerate(self.params.dict_computed_params_names.values())
        }

    @property
    def param_best_dict(self) -> dict:               # Best value of each parameter
        if not hasattr(self, "results") or self.results is None:
            msg = 'No results found. Please run the sampling algorithm first.'
            self._logger.error(msg)
            raise ForMoSAError(msg)

        return {
            name: np.average(self.results['samples'][self.burn_in:, i], weights=self.results['weights'][self.burn_in:])
            for i, name in enumerate(self.params.dict_computed_params_names.values())
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
        return np.average(self.results['logl'][self.burn_in:], weights = self.results['weights'][self.burn_in:])

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

    @property
    def nativ_model(self) -> np.ndarray:             # naiv model
        if not hasattr(self, "results") or self.results is None:
            msg = 'No results found. Please run the sampling algorithm first.'
            self._logger.error(msg)
            raise ForMoSAError(msg)
        return self._nativ_model


    def run(self, results_path: str | os.PathLike, observation: Observation, modelgrid: ModelGrid, interp_method: str = 'linear', wav_fit: list = ['NA'], res_cont: list = ['NA'], bounds_lsq: list = [('NA', 'NA')], emulator: list = ['NA'], full_logL: bool = False) -> None:
        '''
        Method to run the nested sampling algorithm using the model, observation and nested sampling parameters.

        Parameters
        ----------
        logL_type                  (list): Loglikelihood function  (['chi2'], ['chi2_covariance'], ['CCF_Brogi'], ['CCF_Zucker'], ...)
        results_path   (str | os.PathLike): Path of the output
        observation         (Observation): Inbstance of :class:`~ForMoSA.Observation`
        modelGrid           (Observation): Inbstance of :class:`~ForMoSA.ModelGrid`
        interp_method               (str): Interpolation method ('linear', 'cubic', 'spline', ...)
        wav_fit                    (list): List of wavelength grid used for fitting
        res_cont                   (list): List of resolution used for the continuum (used for high contrast)
        bounds_lsq                 (list): List of bounds used for the least squares (used for high contrast)
        emulator                   (list): Emulator of the grid ('PCA', 'NMF')
        full_logL                  (bool): Whether to compute the full loglikelihood function (i.e. with the additional constant noise terms)

        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''
        self._logger.info(f' Run Nested Sampling algorithm using {self.npoints} living points and {self.algorithm}.')

        # Replace 'parX' names by associated physical parameters ('Teff', 'logg', ...)
        for name in self.params.parameters.keys():
            if name.startswith('par'):  # Detect grid parameters
                self.params.parameters[name]._name = modelgrid.titles[modelgrid.keys.index(name)]  # Rename parameter with title associated to 'parX'

        for indobs in range(observation.n_obs):
            if (bounds_lsq[indobs % len(bounds_lsq)] == ('NA', 'NA')) and (not(self.logL[indobs % len(self.logL)].startswith('CCF'))) and (len(observation.obs_data[indobs]['spectro']['star_flx']) > 0):
                msg = f' If you do not chose a CCF mapping loglikelihood function ({self.logL[indobs]}), please chose values for the LSQ bounds'
                self._logger.error(msg)
                raise ForMoSAError(msg)

            obs_data_spectro = observation.obs_data[indobs]['spectro']

            if len(obs_data_spectro['inv_cov']) > 0 and not(self.logL[indobs % len(self.logL)].endswith('_covariance')):
                self._logger.warning(f' observation {observation.obs_name[indobs]} contains a covariance matrix but your loglikelihood function does not account for covariance matrices. You should consider changing the loglikelihood function from {self.logL[indobs]} to {self.logL[indobs] + "_covariance"}.')

            if len(obs_data_spectro['inv_cov']) == 0 and self.logL[indobs % len(self.logL)].endswith('_covariance'):
                msg = f' You chose a loglikelihood accounting for covariance matrices but your observation {observation.obs_name[indobs]} does not contain any covariance matrix. Please adapt your loglikelihood function.'
                self._logger.error(msg)
                raise ForMoSAError(msg)

        n_free_parameters = self._params.n_free_parameters

        loglike_gp = lambda theta: self._loglike(theta, observation, modelgrid, interp_method=interp_method, wav_fit=wav_fit, res_cont=res_cont, bounds_lsq=bounds_lsq, emulator=emulator, full_logL = full_logL)
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

            sampler = ultranest.ReactiveNestedSampler(param_names=self.params.dict_free_params_keys.values(),
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

        # Luminosity derivation
        if r'log(L/L$\mathrm{_{\odot}}$)' in self.params.dict_params_names.values():
            self._results['samples']  = np.column_stack([self._results['samples'], np.zeros(len(self._results['samples']))])
            r_samples = self.param_samples_dict['r']
            Teff_samples = self.param_samples_dict['Teff']

            # Stefan-Boltzmann law
            lum_samples = np.log10(4 * np.pi * (r_samples * cst.R_jup.value) ** 2 * Teff_samples ** 4 * cst.sigma_sb.value / cst.L_sun.value)
            self._results['samples'][:,-1] = lum_samples

        # Mass derivation
        if 'M' in self.params.dict_params_names.values():
            self._results['samples']  = np.column_stack([self._results['samples'], np.zeros(len(self._results['samples']))])

            r_samples = self.param_samples_dict['r']
            logg_samples = self.param_samples_dict['log(g)']

            # Newton law
            M_samples = (r_samples * cst.R_jup.value)**2  / cst.G.value * 10**(logg_samples) / 100 / cst.M_jup.value  # g is in cm/s**2 so we need to convert it in m/s hence the division by 100
            self._results['samples'][:,-1] = M_samples

        self.plotting._ns_results = self.results

        time_elapsed = time.time() - time1
        if time_elapsed < 60:
            time_spent = f'{time_elapsed:.1f} sec'
        elif time_elapsed < 3600:
            time_spent = f'{time_elapsed/60:.1f} min'
        else:
            time_spent = f'{time_elapsed/3600:.1f} hours'

        self._fitted = True

        self._logger.info(f' Time spent: {time_spent}')

        self._logger.info(f'Summary of Nested Sampling : \n {self.summary()}')


    def _loglike(self, theta: list, observation: Observation, modelgrid: ModelGrid, wav_fit: list = ['NA'], res_cont: list = ['NA'], bounds_lsq: list = ['NA'], interp_method: str = 'linear', emulator: list = ['NA'], full_logL : bool = False) -> float | tuple[dict, np.ndarray, np.ndarray]:
        '''
        Compute the loglikelihood for given values of the parameters of the nested sampling

        Parameters
        ----------
        theta                      (list): Parameters values picked by the nested sampling
        observation         (Observation): Inbstance of :class:`~ForMoSA.Observation`
        modelGrid           (Observation): Inbstance of :class:`~ForMoSA.ModelGrid`
        wav_fit                    (list): List of wavelength grid used for fitting
        res_cont                   (list): List of resolution used for the continuum (used for high contrast)
        bounds_lsq                 (list): List of bounds used for the least squares (used for high contrast)
        emulator                   (list): Emulator of the grid ('PCA', 'NMF')
        full_logL                  (bool): Whether to compute the full loglikelihood function (i.e. with additional constant noise terms)

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
                modif_data[indobs], modif_model[indobs] = self._compute_model_from_theta(theta, observation.obs_data[indobs]['spectro'], observation.obs_data[indobs]['photo'], modelgrid.adapted_grid[indobs]['spectro'], modelgrid.adapted_grid[indobs]['photo'], interp_method = interp_method, wav_fit = wav_fit[indobs % len(wav_fit)], res_cont = res_cont[indobs % len(res_cont)], bounds_lsq = bounds_lsq[indobs % len(bounds_lsq)], indobs = indobs)
                # loglike
                logL = self._compute_loglike_from_model_and_spectra(observation.obs_data[indobs]['spectro'], observation.obs_data[indobs]['photo'], modif_model[indobs]['spectro'], modif_model[indobs]['photo'], indobs = indobs, full_logL = full_logL, wav_fit = wav_fit[indobs])

                # Increment total Log-likelihood
                FINAL_logL += logL

        except ForMoSAError as e:
            self._logger.error(f"Error computing loglikelihood: {e}")
            raise ForMoSAError(e)

        return FINAL_logL


    def _compute_model_from_theta(self, theta: list, obs_dict_spectro: dict, obs_dict_photo, grid_spectro: ModelGrid | ModelSubGrid, grid_photo: ModelGrid | ModelSubGrid, interp_method: str = 'linear', wav_fit: str | np.ndarray = 'NA', res_cont: str | np.ndarray = 'NA', bounds_lsq: list = ['NA', 'NA'], indobs: int = 0) -> tuple[dict, dict]:
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
        interp_method                         (str): Method for the interpolation of the grid
        wav_fit                  (str | np.ndarray): Wavelength grid for the continuum estimation of the model (used for high contrast)
        res_cont                 (str | np.ndarray): Resolution of the continuum (used for high contrast)
        bounds_lsq                           (list): Bounds of the least squares estimation (used for high contrast)
        indobs                                (int): Index of the current observation looping

        Returns:
            - obs_dict   (dict): Dictionary of modified spectra {'spectro': dict, 'photo': dict}
            - mod_dict   (dict): Dictionary of modified model {'spectro': dict, 'photo: dict'}

        Author: Simon Petrus, Paulina Palma-Bifani, Allan Denis and Matthieu Ravet
        '''

        # Step 1: Compute the physical model from theta and the grid
        if len(obs_dict_spectro['star_flx']) > 0:
            hc_mode = True
        else:
            hc_mode = False

        mod_dict = self._build_theoretical_model_from_theta(theta, grid_spectro, grid_photo, interp_method='linear', indobs=indobs, hc_mode = hc_mode)

        # Step 2: Apply observational effects (resampling, speckles, scaling if needed)
        obs_dict, mod_dict = self._apply_observation_effects_to_model(mod_dict, obs_dict_spectro, obs_dict_photo, wav_fit=wav_fit, res_cont=res_cont, bounds_lsq=bounds_lsq, indobs=indobs)

        return obs_dict, mod_dict


    def _build_theoretical_model_from_theta(self, theta: list, grid_spectro: ModelGrid | ModelSubGrid, grid_photo: ModelGrid | ModelSubGrid, interp_method: str = 'linear', indobs: int = 0, hc_mode: bool = False) -> dict:
        '''
        Method to compute the theoretical synthetic spectra based only on theta and grid physics.
        It performs interpolation, Doppler shifting, extinction, vsini broadening, CPD addition and scaling.
        This function excludes any resampling or observational effects.

        Parameters
        ----------
        theta                                (list): Parameter values randomly picked by the nested sampling
        grid_spectro     (ModelGrid | ModelSubGrid): Instance of :class:'~ModelGrid' or :class:'~ModelSubGrid' adapted to spectroscopic data
        grid_photo       (ModelGrid | ModelSubGrid): Instance of :class:'~ModelGrid' or :class:'~ModelSubGrid' adapted to photometric data
        interp_method                         (str): Method for the interpolation of the grid
        indobs                                (int): Index of the current observation looping
        hc_mode                              (bool): Whether we are in high-contrast mode

        Returns
        -------
        mod_dict (dict): Dictionary of the synthetic model {'spectro': dict, 'photo': dict}

        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''

        def get_param(name, indobs):
            name = name if name in self.params.parameters else f"{name}_{indobs}" if f"{name}_{indobs}" in self.params.parameters else None
            return None if name is None else self.params._get_param_value(name, theta)

        # Check input theta length
        if len(theta) != self._params.n_free_parameters:
            msg = f"theta length ({len(theta)}) does not match expected number of free parameters ({self._params.n_free_parameters}). The free parameters are {self._params.dict_free_params_names}"
            self._logger.critical(msg)
            raise ForMoSAError(msg)

        # Interpolation parameters from grid
        theta_grid = [get_param(key, indobs)
                      for key in self.params.dict_params_keys.values()
                      if key.startswith('par')]

        # Retrieve model wavelength and resolution

        if grid_spectro is not None:
            wav_mod_spectro, res_mod_spectro, ins_spectro = grid_spectro.wavelength, grid_spectro.resolution, getattr(grid_spectro, 'instrument', 'unknown')
        if grid_photo is not None:
            wav_mod_photo, ins_photo = grid_photo.wavelength, getattr(grid_photo, 'instrument', 'unknown')

        # Interpolate at the values of the grid parameters
        flx_mod_spectro = grid_spectro._interpolate_between_gridpoints(theta_grid, interp_method, indobs)
        flx_mod_photo = grid_photo._interpolate_between_gridpoints(theta_grid, interp_method, indobs)

        # Save native model before any transformation
        flx_mod_spectro_nativ = np.copy(flx_mod_spectro)
        wav_mod_spectro_nativ = np.copy(wav_mod_spectro)

        # Doppler shifting
        rv = get_param('rv', indobs)
        if rv is not None:
            wav_mod_spectro, flx_mod_spectro, res_mod_spectro = us.doppler_fct(wav_mod_spectro, flx_mod_spectro, res_mod_spectro, rv)

        # vsini correction
        vsini = get_param('vsini', indobs)
        ld = get_param('ld', indobs)
        if vsini is not None and ld is not None:
            try:
                vsini_function = str(self.params.parameters['vsini'].vsini_function)
            except KeyError:
                vsini_function = str(self.params.parameters[f'vsini_{indobs}'].vsini_function)
            flx_mod_spectro, res_mod_spectro = us.vsini_fct(wav_mod_spectro, flx_mod_spectro, res_mod_spectro, ld, vsini, vsini_function)

        # Reddening
        av = get_param('av', indobs)
        if av is not None:
            flx_mod_spectro = us.reddening_fct(wav_mod_spectro, flx_mod_spectro, av)
            flx_mod_photo = us.reddening_fct(wav_mod_photo, flx_mod_photo, av)

        # CPD contribution
        bb_T = get_param('bb_T', indobs)
        bb_R = get_param('bb_R', indobs)
        d = get_param('d', indobs)
        if None not in (bb_T, bb_R, d):
            flx_mod_spectro = us.bb_cpd_fct(wav_mod_spectro, flx_mod_spectro, d, bb_T, bb_R)
            flx_mod_photo = us.bb_cpd_fct(wav_mod_photo, flx_mod_photo, d, bb_T, bb_R)

        # Scaling (ck) via R^2/d^2
        alpha = get_param('alpha', indobs)
        if alpha is None:
            alpha = 1
        r = get_param('r', indobs)
        if r is not None and d is not None and not(hc_mode):
            flx_mod_spectro, ck_spectro = us.calc_ck(flx_mod_spectro, np.array([]), np.array([]), r, d, alpha)
            flx_mod_photo, ck_photo = us.calc_ck(flx_mod_photo, np.array([]), np.array([]), r, d, alpha)
        else:
            ck_spectro, ck_photo = 1, 1  # Will be set observationally if needed

        return {
            'spectro': {
                'wav': wav_mod_spectro,
                'flx': flx_mod_spectro,
                'nativ_flx': flx_mod_spectro_nativ,
                'nativ_wav': wav_mod_spectro_nativ,
                'res': res_mod_spectro,
                'ins': ins_spectro,
                'ck': ck_spectro
            },
            'photo': {
                'wav': wav_mod_photo,
                'flx': flx_mod_photo,
                'ins': ins_photo,
                'ck': ck_photo
            }
        }


    def _apply_observation_effects_to_model(self, model_dict: dict, obs_dict_spectro: dict, obs_dict_photo: dict, wav_fit='NA', res_cont='NA', bounds_lsq=['NA','NA'], indobs=0) -> tuple[dict, dict]:
        '''
        Apply effects specifics to observations: resolution resampling, speckle subtraction, and ck estimation when no physical scaling is defined.

        Parameters
        ----------
        model_dict                          (dict): Model spectra and photometry as built from theta
        obs_dict_spectro                    (dict): Dictionary of spectroscopic observations
        obs_dict_photo                      (dict): Dictionary of photometric observations
        wav_fit                   (str | np.ndarray): Wavelength grid for the continuum estimation of the model (used for high contrast)
        res_cont                  (str | np.ndarray): Resolution of the continuum (used for high contrast)
        bounds_lsq                          (list): Bounds of the least squares estimation (used for high contrast)
        indobs                               (int): Index of the current observation looping

        Returns
        -------
        obs_dict     (dict): update observations dictionary
        model_dict   (dict): updated and model dictionary

        Author: Simon Petrus, Matthieu Ravet and Allan Denis
        '''

        contributions = 1

        # Resolution decreasing and resampling
        if len(model_dict['spectro']['wav']) != len(obs_dict_spectro['wav']):
            model_dict['spectro']['flx'] = us.resolution_decreasing(model_dict['spectro']['wav'], model_dict['spectro']['flx'], model_dict['spectro']['res'], obs_dict_spectro['wav'], obs_dict_spectro['res'])

        # High contrast modeling if stellar flux is available
        if len(obs_dict_spectro['star_flx']) > 0:

            flx_cont_mod = us.continuum_estimate(obs_dict_spectro['wav'], model_dict['spectro']['flx'] * obs_dict_spectro['transm'], model_dict['spectro']['res'], wav_fit, res_cont)

            if self.logL[indobs % len(self.logL)].startswith('chi2'):
                contributions, model_dict['spectro']['flx'], obs_dict_spectro['speckles'], obs_dict_spectro['estimated_system'] = hc._hc_model_estimate_speckles(obs_dict_spectro['flx'], obs_dict_spectro['flx_cont'], obs_dict_spectro['transm'], obs_dict_spectro['star_flx'], obs_dict_spectro['star_flx_cont'], model_dict['spectro']['flx'], flx_cont_mod, obs_dict_spectro['err'], bounds_lsq, obs_dict_spectro['system'])
            else:
                _, model_dict['spectro']['flx'], obs_dict_spectro['speckles'] = hc._hc_model_remove_speckles(obs_dict_spectro['flx'], obs_dict_spectro['flx_cont'], obs_dict_spectro['transm'], obs_dict_spectro['star_flx'], obs_dict_spectro['star_flx_cont'], model_dict['spectro']['flx'], flx_cont_mod, obs_dict_spectro['err'])
                obs_dict_spectro['estimated_system'] = np.repeat(0, len(obs_dict_spectro['wav']))
        else:
            obs_dict_spectro['speckles'], obs_dict_spectro['estimated_system'] = np.repeat(0, len(obs_dict_spectro['wav'])), np.repeat(0, len(obs_dict_spectro['wav']))

        # Optional analytical ck when no r/d available
        if model_dict['spectro']['ck'] == 1 and len(obs_dict_spectro['star_flx']) == 0:
            model_dict['spectro']['flx'], ck_spectro = us.calc_ck(model_dict['spectro']['flx'], obs_dict_spectro['flx'], obs_dict_spectro['err'], 0, 0, alpha=0, analytic='yes')
            model_dict['photo']['flx'], ck_photo = us.calc_ck(model_dict['photo']['flx'], obs_dict_photo['flx'], obs_dict_photo['err'], 0, 0, alpha=0, analytic='yes')
            model_dict['spectro']['ck'] = ck_spectro
            model_dict['photo']['ck'] = ck_photo

        # Update info
        model_dict['spectro'].update({'wav': obs_dict_spectro['wav'], 'res': obs_dict_spectro['res'], 'hc_contributions': contributions})

        obs_dict = {'spectro': obs_dict_spectro,
                    'photo': obs_dict_photo}

        return obs_dict, model_dict


    def _compute_loglike_from_model_and_spectra(self, obs_dict_spectro: dict, obs_dict_photo: dict, mod_dict_spectro: dict, mod_dict_photo: dict, indobs: int = 0, full_logL: bool = False, wav_fit: np.ndarray | float = 'NA'):
        '''
        Method to compute the loglikelihood from the modified observation and model

        Parameters
        ----------
        obs_dict_spectro    (dict): Dictionary of observation spectroscopic data modified by the nested sampling {'wav': wav, 'flx': flx, ...}
        obs_dict_photo      (dict): Dictionary of observation photometric data modified by the nested sampling {'wav': wav, 'flx': flx, ...}
        mod_dict_spectro    (dict): Dictionary of model spectroscopic data modified by the nested sampling {indobs: {'spectro': dict 'photo': dict}}
        mod_dict_photo      (dict): Dictionary of model photometric data modified by the nested sampling {indobs: {'spectro': dict 'photo': dict}}
        indobs               (int): Index of current observation
        full_logL           (bool): Whether to compute the full loglikelihood function (i.e. with additional constant noise terms)
        wav_fit (str | np.ndarray): Wavelengths used for fitting

        Returns:
            - Final_logL (float): Final loglikelihood value

        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''

        # LogL Photometry
        if len(obs_dict_photo['wav']) == 0:
            logL_photo = 0
        else:
            residual_photo = obs_dict_photo['flx'] - mod_dict_photo['flx']
            logL_photo = logL_functions.logL_chi2(residual_photo, obs_dict_photo['err'], full=full_logL)

        # LogL Spectroscopy
        logL_spectro = 0

        if len(obs_dict_spectro['wav']) > 0:
            # Bounds for fitting
            if wav_fit != 'NA':
                ind_for_fitting = np.array([], dtype=int)
                for wav_fit_cut in wav_fit.split('/'):
                    wmin, wmax = map(float, wav_fit_cut.split(','))
                    indices = np.where((obs_dict_spectro['wav'] >= wmin) & (obs_dict_spectro['wav'] <= wmax))[0]
                    ind_for_fitting = np.concatenate((ind_for_fitting, indices))
                ind_for_fitting = np.sort(ind_for_fitting)
            else:
                ind_for_fitting = np.arange(len(obs_dict_spectro['wav']))

            # Selection of final products for the loglikelihood computation
            obs_flx = obs_dict_spectro['flx'][ind_for_fitting]
            mod_flx = mod_dict_spectro['flx'][ind_for_fitting]
            speckles = obs_dict_spectro['speckles'][ind_for_fitting]
            system = obs_dict_spectro['estimated_system'][ind_for_fitting]
            err = obs_dict_spectro['err'][ind_for_fitting]

            residual = obs_flx - mod_flx - speckles - system

            # Specific case for covariance matrix
            if len(obs_dict_spectro['cov'] > 0):
                cov = obs_dict_spectro['cov'][np.ix_(ind_for_fitting, ind_for_fitting)]

            # Same with inverse of covariance matrix
            if len(obs_dict_spectro['inv_cov']) > 0:
                inv_cov = obs_dict_spectro['inv_cov'][np.ix_(ind_for_fitting, ind_for_fitting)]

            # Loglikelihood type
            ll_type = self.logL[indobs % len(self.logL)]

            logL_dict = {
                'chi2': lambda: logL_functions.logL_chi2(residual, err),
                'chi2_covariance': lambda: logL_functions.logL_chi2_covariance(residual, cov, inv_cov, full=full_logL),
                'CCF_Brogi': lambda: logL_functions.logL_CCF_Brogi(obs_flx - speckles - system, mod_flx),
                'CCF_Zucker': lambda: logL_functions.logL_CCF_Zucker(obs_flx - speckles - system, mod_flx),
                'CCF_custom': lambda: logL_functions.logL_CCF_custom(obs_flx - speckles - system, mod_flx, err),
                'chi2_noisescaling': lambda: logL_functions.logL_chi2_noisescaling(residual, err, full=full_logL),
                'chi2_noisescaling_covariance': lambda: logL_functions.logL_chi2_noisescaling_covariance(residual, cov, inv_cov, full=full_logL)
            }

            logL_spectro = logL_dict.get(ll_type, lambda: 0)()

        # LogL final
        FINAL_logL = logL_photo + logL_spectro


        if FINAL_logL < -1e6:
            self._logger.warning(f"[loglike WARNING] Unusually low loglike: {FINAL_logL}")
            for name in self.params.free_parameters:
                value = self.params._get_param_value(name, list(self.params.theta.values()))
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

        Returns:
            - prior   (list): List of parameter values transformed by the prior laws, in the original order

        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''

        prior = dict()
        theta_index_free = self.params.free_parameters.keys()   # List of free (without constant priors) parameters
        theta_index_fixed = self.params.fixed_parameters.keys()

        for i, param_name in enumerate(theta_index_free):
            param = self.params.parameters[param_name]
            theta_val = theta[i]
            prior_value = param._apply_prior(theta_val)

            if param_name.startswith('par'):
                # Clamp within the grid bounds
                prior_value = max(min(prior_value, modelGrid.lims_params_grid[param_name][1]), modelGrid.lims_params_grid[param_name][0])
                param._theta = prior_value

            prior[param_name] = prior_value

        prior_free = list(prior.values())

        for param_name in theta_index_fixed:
            param = self.params.parameters[param_name]
            value = param.value

            prior[param_name] = value

        self.params._theta = prior    # Update the current drawn values for the parameters

        return prior_free


    def _compute_best_model(self, observation: Observation, modelgrid: ModelGrid, interp_method: str = 'linear', wav_fit: list = ['NA'], res_cont: list = ['NA'], bounds_lsq: list = ['NA', 'NA']) -> None:
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

        best_theta = self.list_best_params[:self.params.n_free_parameters]

        modif_data, best_model = dict(), dict()

        for indobs in range(observation.n_obs):
            modif_data[indobs], best_model[indobs] = self._compute_model_from_theta(best_theta, observation.obs_data[indobs]['spectro'], observation.obs_data[indobs]['photo'], modelgrid.adapted_grid[indobs]['spectro'], modelgrid.adapted_grid[indobs]['photo'], interp_method = interp_method, wav_fit = wav_fit[indobs % len(wav_fit)], res_cont = res_cont[indobs % len(res_cont)], bounds_lsq = bounds_lsq[indobs % len(bounds_lsq)], indobs = indobs)

        self._modif_data = modif_data
        self._best_model = best_model


    def _compute_nativ_model_from_theta(self, theta: list, nativ_grid: ModelGrid, observation: Observation, wavelength_range: list | str = 'obs', resolution: float | str = 'nativ'):
        '''
        Method to compute the nativ grid

        Parameters
        ----------
        theta                      (list): List of parameters to compute the theoretical model
        nativ_grid            (ModelGrid): Instance of :class:~'ModelGrid'
        observation         (Observation): Instance of :class:~'Observation'
        wavelength_range     (list | str): Wavelength range to compute the nativ model to
        resolution          (float | str): Resolution to decrease the nativ model to.

        Returns
            - nativ_model: Nativ model transformed by the list of parameters

        Authors: Allan Denis
        '''

        if wavelength_range != 'obs':
            cut = (nativ_grid.wavelength <= wavelength_range[1]) & (nativ_grid.wavelength >= wavelength_range[0])
            target_wavelength = nativ_grid.wavelength[cut]
        else:
            cut = (nativ_grid.wavelength <= observation.wavelength_range[0]) & (nativ_grid.wavelength >= observation.wavelength_range[1])
            target_wavelength = nativ_grid.wavelength[cut]

        target_resolution = observation.min_resolution
        indices = nativ_grid._find_valid_resolution_region(nativ_grid.wavelength, nativ_grid.resolution, target_wavelength, target_resolution)
        nativ_grid.grid = nativ_grid.grid.isel(wavelength=indices)
        nativ_grid.grid.attrs['res'] = nativ_grid.grid.attrs['res'][indices]
        nativ_grid._resolution = nativ_grid._resolution[indices]
        nativ_model = self._build_theoretical_model_from_theta(theta, nativ_grid, ModelSubGrid('', nativ_grid.grid, self._logger, [], [], 'photo'))

        if resolution == 'nativ':
            resolution = nativ_model['spectro']['res']
        elif resolution == 'obs':
            resolution = np.full(len(nativ_model['spectro']['res']), observation.min_resolution)
        else:     # resolution is a float
            resolution = np.full(len(nativ_model['spectro']['res']), resolution)

        nativ_model_flx = us.resolution_decreasing(nativ_model['spectro']['wav'], nativ_model['spectro']['flx'], nativ_model['spectro']['res'], nativ_model['spectro']['wav'], resolution)
        nativ_model['spectro'].update({'wav': nativ_model['spectro']['wav'], 'flx': nativ_model_flx, 'res': resolution})

        return nativ_model


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


        self.plotting._ns_results = self.results
        self.plotting._list_params = list(self.param_best_dict.keys())


    def summary(self, sigma: int = 2, LaTeX: bool = False) -> None:
        '''
        Method to print a summary of the nested sampling results including weighted statistics.

        Parameters
        ----------
        sigma     (int): Confidence interval (1 or 2 sigma), default is 1.
        LaTeX    (bool): Whether to format the output for LaTeX.

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

        def weighted_percentile(p):
            return np.interp(p / 100, cumsum_weights, sorted_samples)

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

            low = weighted_percentile(low_pct)
            high = weighted_percentile(high_pct)

            plus  = high - mean
            minus = low - mean
            msg += f" {list(self.params.dict_computed_params_names.values())[i]:10s}: {mean:10.4f} {minus:+10.4f} {plus:+10.4f} [{low:10.4f}, {high:10.4f}] ({sigma}σ)\n"

        msg += "=========================================\n"

        if LaTeX:
            for i in range(samples.shape[1]):
                param_samples = samples[:, i]

                # Weighted mean
                mean = np.average(param_samples, weights=weights)

                # Weighted percentiles
                sorted_indices = np.argsort(param_samples)
                sorted_samples = param_samples[sorted_indices]
                sorted_weights = weights[sorted_indices]
                cumsum_weights = np.cumsum(sorted_weights)

                low = weighted_percentile(low_pct)
                high = weighted_percentile(high_pct)

                plus  = high - mean
                minus = low - mean
                msg += f" {list(self.params.dict_computed_params_names.values())[i]:10s}: ${mean:.4f}_{{{minus:+.4f}}}^{{{plus:+.4f}}}$\n"

            msg += "=========================================\n"

        return msg




