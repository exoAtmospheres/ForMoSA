import logging
import ForMoSA.utils.misc as utils
import ForMoSA.utils.spec as spec
import colorlog
from ForMoSA.global_params import GlobalParams
from ForMoSA.model_grid import ModelGrid
from ForMoSA.paths import ForMoSAPaths
from ForMoSA.observation import Observation
from ForMoSA.nested_sampling.sampling import NestedSampling
from ForMoSA.nested_sampling.plotting import NestedSamplingPlotting
from ForMoSA.nested_sampling.parameters import Parameter
from ForMoSA.error import ForMoSAError
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from matplotlib import font_manager
font_path = '/Users/rajpoot/Library/Fonts/JuliaMono-Regular.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

# set font size to 18
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

# log
_log = logging.getLogger(__name__)

class Analysis(object):
    '''
    ForMoSA data analysis class

    Parameters
    ----------
    global_params                 (dict): Dictionary of global parameters
    adapted                       (bool): Whether the model is adapted to the data, by default False. Can be set to True if the model has already been adapted to the data
    fitted                        (bool): Whether the data have already been fitted for
    log_level (str): og level of the handler, by default ``'info'`` for all important informations.

    Authors: Allan Denis
    '''

    def __init__(self, global_params: GlobalParams, adapted: bool = False, fitted: bool = False, log_level: str = 'info') -> None:

        logger = logging.getLogger("ForMoSA")

        logger.propagate = False

        while logger.hasHandlers():
            logger.removeHandler(logger.handlers[0])

        logger.setLevel(log_level.upper())

        # File handler (no color)
        file_handler = logging.FileHandler(global_params.paths._result_path / 'analysis.log', mode='w', encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s\t%(levelname)8s\t%(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler (with color)
        console_handler = colorlog.StreamHandler()
        console_formatter = colorlog.ColoredFormatter(
            fmt='%(log_color)s[%(levelname)s] %(message)s',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Inits
        self._config = global_params.config
        self._paths = global_params.paths
        self._adapted = adapted
        self._fitted = fitted
        self._ns = NestedSampling(self.config_params['inversion']['ns_algo'], self.config_params['inversion']['npoints'], self.config_params['inversion']['logL_type'], logger, self.config_params['ns_algo'])
        self._logger = logger
        # Build and check list of nested sampling parameters
        self.ns.params._add_NestedSampling_parameters_from_config(self.config_params['parameters'])

        # Replace 'parX' names by associated physical parameters ('Teff', 'logg', ...)
        for name in self.ns.params.parameters.keys():
            if name.startswith('par'):  # Detect grid parameters
                self.ns.params.parameters[name]._name = self.grid.titles[self.grid.keys.index(name)]  # Rename parameter with title associated to 'parX'

        # Luminosity
        if ('Teff' in self.ns.params.dict_free_params_keys.values()) and ('r' in self.ns.params.dict_free_params_names.values()):
            lum_param = Parameter(r'log(L/L$\mathrm{_{\odot}}$)', 'computed')
            self.ns.params._add_parameter(lum_param, lum_param.name)

        # Mass
        if ('log(g)' in self.ns.params.dict_free_params_names.values()) and ('r' in self.ns.params.dict_free_params_names.values()):
            Mass_param = Parameter('M', 'computed')
            self.ns.params._add_parameter(Mass_param, Mass_param.name)

        self._ns._plotting = NestedSamplingPlotting(logger, self.config_params['plottings'], list_params=list(self.ns.params.dict_computed_params_names.values()))

        adapt = self.config_params['adapt']
        inversion = self.config_params['inversion']

        res_obs   = adapt['target_res_obs']
        res_mod   = adapt['target_res_mod']
        res_cont  = adapt['res_cont']
        emulator  = adapt['emulator']
        wav_fit   = inversion['wav_fit']
        logL_type = inversion['logL_type']

        # Parameters we want to check the format
        params = {
            'res_obs': res_obs,
            'res_mod': res_mod,
            'res_cont': res_cont,
            'emulator': emulator,
            'wav_fit': wav_fit,
            'logL_type': logL_type
        }

        n_obs = self.observation.n_obs

        wrong_type = [name for name, val in params.items() if not isinstance(val, list)]
        wrong_length = [name for name, val in params.items() if not (len(val) == 1 or len(val) == n_obs)]

        # Errors
        if wrong_type:
            msg = f"Params not list: {', '.join(wrong_type)}."
            self._logger.critical(msg)
            raise ForMoSAError(msg)

        if wrong_length:
            msg = f"Params with wrong length (must be 1 or {n_obs}): {', '.join(wrong_length)}."
            self._logger.critical(msg)
            raise ForMoSAError(msg)

        if self.adapted:
            # Load adapted observations and grids
            try:
                self.observation._load_adapted_observations_from_files(self.paths.result_path)
            except ForMoSAError:
                self._logger.warning(f' Adapting and saving observations to folder {self.paths.result_path}')
                self.observation.adapt_all_observations(res_obs, self.grid.wavelength, self.grid.resolution, res_cont = res_cont, wav_cont = wav_fit)
                self.observation._save_all_observations(self.paths.result_path)

            self.grid._load_grid_from_files(self.paths.adapt_store_path, self.observation.obs_name_list)
        if self.fitted:
            # Load results
            self.ns._load_results(self.paths.result_path)


    ##################################################
    # Representation
    ##################################################

    def __repr__(self) -> str:
        return f'<Analysis, config_file={self.paths.config_file_path}>'

    def __format__(self) -> str:
        return self.__repr__()

    ##################################################
    # Properties
    ##################################################

    @property
    def loglevel(self) -> str:
        return logging.getLevelName(self._logger.level)

    @loglevel.setter
    def loglevel(self, level: str):
        self._logger.setLevel(level.upper())

    @property
    def adapted(self) -> bool:
        return self._adapted

    @adapted.setter
    def adapted(self, adapted_status: bool) -> bool:
        self._adapted = adapted_status

    @property
    def ns(self) -> NestedSampling:
        return self._ns

    @property
    def config_params(self) -> dict:
        return self._config

    @property
    def paths(self) -> ForMoSAPaths:
        return self._paths

    @property
    def observation(self) -> Observation:
        return self.paths.observation

    @property
    def grid(self) -> ModelGrid:
        return self.paths.grid

    @property
    def fitted(self) -> bool:
        return self._fitted

    @fitted.setter
    def fitted(self, fitted_status: bool) -> bool:
        self._fitted = fitted_status


    ##################################################
    # Methods
    ##################################################


    def adapt(self):
        '''
        Method to adapt the grid of model to each observation

        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''

        adapt = self.config_params['adapt']
        inversion = self.config_params['inversion']

        lower_bounds_lsq = inversion['hc_lower_bounds_lsq']
        higher_bounds_lsq = inversion['hc_higher_bounds_lsq']
        bounds_lsq = [(low_bound, high_bound) for low_bound, high_bound in zip(lower_bounds_lsq, higher_bounds_lsq)]

        res_obs   = adapt['target_res_obs']
        res_mod   = adapt['target_res_mod']
        res_cont  = adapt['res_cont']
        emulator  = adapt['emulator']
        wav_fit   = inversion['wav_fit']
        full_logL     = inversion['full_logL']

        # Parameters we want to check the format
        params = {
            'res_obs': res_obs,
            'res_mod': res_mod,
            'res_cont': res_cont,
            'emulator': emulator,
            'wav_fit': wav_fit,
            'bounds_lsq': bounds_lsq,
            'full_logL': full_logL
        }

        n_obs = self.observation.n_obs

        wrong_type = [name for name, val in params.items() if name != 'full_logL' and not isinstance(val, list)]
        wrong_length = [name for name, val in params.items() if name != 'full_logL' and not (len(val) == 1 or len(val) == n_obs)]

        # Errors
        if wrong_type:
            msg = f"Params not list: {', '.join(wrong_type)}."
            self._logger.critical(msg)
            raise ForMoSAError(msg)

        if wrong_length:
            msg = f"Params with wrong length (must be 1 or {n_obs}): {', '.join(wrong_length)}."
            self._logger.critical(msg)
            raise ForMoSAError(msg)

        self.observation.adapt_all_observations(res_obs, self.grid.wavelength, self.grid.resolution, res_cont = res_cont, wav_cont = wav_fit)

        if not self.adapted:   # If the model is not already adapted to the data, or if the user wants to redo the adaptation
            # Adapt grid using target wavelength and resolution
            self.grid.adapt_all_grids(self.observation.obs_data, res_mod, self.ns.params, wav_cont = wav_fit, res_cont = res_cont)

            if emulator == 'PCA':
                self._logger.info(' Decomposing the grid using PCA')

                # TODO

            if emulator == 'NMF':
                self._logger.info(' Decomposing the grid using NMF')

                # TODO

            # Save the data
            self.observation._save_all_observations(self.paths.result_path)

            self.grid._interpolate_missing_values()
            self.grid._save_grid(self.paths.adapt_store_path)

            # grid and data are now adapted
            self._adapted = True


    def nested_sampling(self):
        '''
        Method to launch the nested sampling algorithm

        Parameters
        ----------
        algorithm        (str): Algorithm to be used ('nestle', 'ultranest', 'pymultinest')
        logL_function   (list): logL type for each observation
        wav_for_fitting (list): Wavelength grid used for fitting for each observation
        wav_cont        (list): Wavelength grid used for the continuum for each observation
        res_cont        (list): Resolution used for the continuum for each observation
        bounds_lsq      (list): Least Squares bounds used for each observation (for high-contrast observations only)
        interp_method    (str): Interpolation method to interpolate between the gridpoints

        Authors: Allan Denis
        '''

        adapt = self.config_params['adapt']
        inversion = self.config_params['inversion']

        res_cont      = adapt['res_cont']
        emulator      = adapt['emulator']
        interp_method = adapt['method']

        wav_fit       = inversion['wav_fit']
        full_logL     = inversion['full_logL']

        lower_bounds_lsq = inversion['hc_lower_bounds_lsq']
        higher_bounds_lsq = inversion['hc_higher_bounds_lsq']
        bounds_lsq = [(low_bound, high_bound) for low_bound, high_bound in zip(lower_bounds_lsq, higher_bounds_lsq)]

        if not(self.fitted):
            # Run nested sampling
            self.ns.run(self.paths.result_path, self.observation, self.grid, interp_method=interp_method, res_cont=res_cont, bounds_lsq=bounds_lsq, emulator=emulator, full_logL = full_logL, wav_fit = wav_fit)

            # Savings
            self.ns._save_results(self.paths.result_path)
            self._fitted = True

        # Best model cut at the wavelengths of the observations
        self.ns._compute_best_model(self.observation, self.grid, interp_method = interp_method, wav_fit = wav_fit, res_cont = res_cont, bounds_lsq = bounds_lsq)
        self.grid._read_grid()
        # Best nativ model
        self.ns._nativ_model = self.ns._compute_nativ_model_from_theta(self.ns.list_best_params[:self.ns.params.n_free_parameters], self.grid, self.observation, wavelength_range=self.grid.wavelength_range, resolution=self.observation.min_resolution)


    def plot(self, label_ins: str = 'no', trans: str = 'yes', uncert: str = 'yes', figsize_corner: tuple = (15, 15), figsize_chains: tuple = (12, 15), figsize_fit: tuple = (20, 10), save: bool = True, plot_nativ_model: bool = False, quantiles: list = [16, 50, 84], label_params: bool = True) -> None:
        '''
        Method to use all the plotting methods

        Parameters
        ----------
        label_ins         (str): Whether to label instruments in best fit plot
        trans             (str): Whether to plot the transmission filters
        uncert            (str): Whether to plot the uncertainties
        save              (bool): Whether to save the plots, by default True
        plot_nativ_model  (bool): Whether to plot the nativ model
        quantiles         (list): Quantiles to use in the radar plot and the nativ model uncertainties

        Authors: Allan Denis and Arthur Vigan
        '''

        results = self.ns.results
        modif_data = self.ns.modif_data
        best_model = self.ns.best_model
        nativ_model = self.ns.nativ_model

        all_nativ_models = []

        if plot_nativ_model:
            self._logger.info('Compute nativ model for each sample')

            samples = results["samples"][:,:self.ns.params.n_free_parameters]
            weights = results["weights"]

            n_keep = 1000  # Keep maximum 1000 samples

            if len(samples) > n_keep:
                indices = np.random.choice(len(samples), n_keep, replace=False)
                samples = samples[indices]
                weights = weights[indices]

            for theta in tqdm(samples):
                nativ_model_i = self.ns._compute_nativ_model_from_theta(theta, self.grid, self.observation, wavelength_range=self.grid.wavelength_range, resolution=self.observation.min_resolution)
                all_nativ_models.append(nativ_model_i['spectro']['flx'])

            all_nativ_models = np.array(all_nativ_models)   # shape = (n_samples, n_wavelengths)

            lower_1sigma = utils.get_weighted_percentile(quantiles[0], all_nativ_models, weights=weights)
            upper_1sigma = utils.get_weighted_percentile(quantiles[-1], all_nativ_models, weights=weights)

        fig = self.ns.plotting.plot_corner(figsize=figsize_corner, show_titles = True, show_contours = True, plot_density = True, quantiles = [0.16, 0.5, 0.84])
        if save:
            filename = self.paths.result_path / 'corner_plot.pdf'
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            self._logger.info(f"Corner plot saved to {filename}")

        fig, _ = self.ns.plotting.plot_chains(figsize=figsize_chains)
        if save:
            filename = self.paths.result_path / 'chains_plot.pdf'
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            self._logger.info(f"Chains plot saved to {filename}")

        fig, _ = self.ns.plotting.plot_radar()
        if save:
            filename = self.paths.result_path / 'radar_plot.pdf'
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            self._logger.info(f'Radar plot saved to {filename}')

        fig, ax, _, _, scaling = self.ns.plotting.plot_fit(modif_data, best_model, label_ins=label_ins, trans=trans, uncert=uncert, figsize=figsize_fit, plot_nativ_model = plot_nativ_model, nativ_model = nativ_model, label_params = label_params)
        if plot_nativ_model:
            ax.fill_between(nativ_model['spectro']['wav'], scaling * lower_1sigma, scaling * upper_1sigma, color = 'grey', alpha = 0.4)
        if save:
            filename = self.paths.result_path / 'best_fit_plot.pdf'
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            self._logger.info(f"Best fit plot saved to {filename}")


    def plot_ccf(self, rv_grid: np.ndarray, indobs: int = 0, save: bool = True, plot: bool = True, theta: dict = dict(), zoom: bool = True) -> None:
        '''
        Method to compute (and optionally plot) the cross-correlation function (CCF) between the observation and the best model.

        Parameters
        ----------
        rv_grid       (array): Grid of radial velocities to evaluate the CCF over
        indobs          (ind): Index of the observation (default is 0)
        save           (bool): Whether to save the plot as a file
        plot           (bool): Whether to display the plot
        theta   (dictionnary): Dictionnary of the parameters to use for the model
        zoom          (bool): Whether to zoom in on the CCF plot around the peak

        Returns
        -------
        ccf (array): Crosscorrelation function
        acf (array): Autocorrelation function

        Authors: Allan Denis
        '''
        if len(theta) == 0:
            self._logger.info('No value provided for the list of parameters of the model. Using best vale from the Nested Sampling')
            if not self._fitted:
                raise ForMoSAError("Nested sampling must be run before computing the CCF.")# Best params to compute the best model
            best_params = self.ns.param_best_dict.copy()
            vsini_index = list(best_params.keys()).index('vsini')
            best_params['rv'] = 0
            theta = list(best_params.values())
        else:
            vsini_index = list(theta.keys()).index('vsini')
            theta['rv'] = 0
            theta = list(theta.values())

        self._logger.info(f'Computing CCF for observation {self.observation.obs_name[indobs]}...')

        adapt = self.config_params['adapt']
        inversion = self.config_params['inversion']

        res_cont = float(adapt['res_cont'][indobs % len(adapt['res_cont'])])
        wav_fit = inversion['wav_fit'][indobs % len(inversion['wav_fit'])]

        lower_bounds_lsq = inversion['hc_lower_bounds_lsq']
        higher_bounds_lsq = inversion['hc_higher_bounds_lsq']
        bounds_lsq = (lower_bounds_lsq[indobs], higher_bounds_lsq[indobs])

        grid_spectro, grid_photo = self.grid.adapted_grid[indobs]['spectro'], self.grid.adapted_grid[indobs]['photo']
        obs_dict_spectro, obs_dict_photo = self.observation.obs_data[indobs]['spectro'], self.observation.obs_data[indobs]['photo']

        res_mod_spectro = grid_spectro.grid.attrs['res']
        res_obs_spectro = obs_dict_spectro['res']

        # Best model
        _, mod_dict_spectro = self.ns._compute_model_from_theta(theta, obs_dict_spectro, obs_dict_photo, grid_spectro, grid_photo, res_mod_spectro, wav_fit=wav_fit, res_cont=res_cont, bounds_lsq=bounds_lsq)
        vsini_fct = self.ns.params.parameters['vsini'].vsini_function

        # Using nativ_flx which the flux at the nativ resolution of the model
        wav_mod = mod_dict_spectro['spectro']['nativ_wav']
        flx_mod = mod_dict_spectro['spectro']['nativ_flx']

        wav_obs = obs_dict_spectro['wav']
        flx_obs = obs_dict_spectro['flx']
        err_obs = obs_dict_spectro['err']
        star_flx = obs_dict_spectro['star_flx']
        transm = obs_dict_spectro['transm']
        system = obs_dict_spectro['system']

        flx_mod_vsini, res_mod_vsini = spec.vsini_fct(wav_mod, flx_mod, res_mod_spectro, 0.6, theta[vsini_index], vsini_fct)

        # CCF
        ccf, acf, ccf_star, rv_peak, _, _ = spec.compute_ccf(wav_mod, flx_mod_vsini, wav_obs, flx_obs, err_obs, res_mod_vsini, res_obs_spectro, res_cont, wav_fit, star_flx, transm, system, rv_grid=rv_grid)

        # Plot
        fig = plt.figure('CCF', figsize=(12,6), dpi=300)
        plt.clf()
        ax = fig.add_subplot()
        ax.plot(rv_grid, ccf, color='C0', label='CCF')
        ax.plot(rv_grid, ccf_star, color='0.85', zorder=-1000, label='Speckles')
        ax.plot(rv_grid + rv_peak, acf, 'k', label = 'Auto-correlation')
        ax.axvline(x=rv_peak, linestyle='--', c='C3')
        ax.set_xlabel('Radial Velocity [km/s]')
        ax.set_ylabel('Correlation (SNR)')
        # ax.set_title(f'Cross-Correlation Function - Observation {self.observation.obs_name[indobs]}')

        # enable legend
        ax.legend(fontsize=12)

        #let's write a more informative title with 
        param_str = ', '.join([f"{name}={value:.2f}" for name, value in zip(self.ns.params.dict_free_params_names.values(), theta)])
        ax.set_title(f'CCF - Obs: {self.observation.obs_name[indobs]}\nTeff={theta[0]:.1f} K,log(g)={theta[1]:.1f}, RV={rv_peak:.1f} km/s, vsini={theta[vsini_index]:.1f} km/s')

        ax.grid(True)
        if save:
            filename = self.paths.result_path / f'ccf_plot_obs{indobs}.pdf'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            self._logger.info(f"CCF plot saved to {filename}")
        if plot:
            plt.show()
        if zoom:
            rv_min = rv_peak - 500
            rv_max = rv_peak + 500
            ax.set_xlim(rv_min, rv_max)
            plt.savefig(self.paths.result_path / f'ccf_plot_obs{indobs}_zoom.pdf', dpi=300, bbox_inches='tight')
            self._logger.info(f"Zoomed CCF plot saved to {self.paths.result_path / f'ccf_plot_obs{indobs}_zoom.pdf'}")


        return fig, ax, rv_peak, ccf, acf


    def plot_rv_vsini_map(self, rv_grid: np.ndarray, vsini_grid: np.ndarray, indobs: int = 0, save: bool = True, plot: bool = True, theta: dict = dict(), bounds: tuple = (-np.inf, np.inf)) -> None:
        '''
        Method to compute (and optionally plot) the rv / vsini map between the observation and the best model.

        Parameters
        ----------
        rv_grid       (array): Grid of radial velocities to evaluate the CCF over
        indobs          (ind): Index of the observation (default is 0)
        save           (bool): Whether to save the plot as a file
        plot           (bool): Whether to display the plot
        theta   (dictionnary): Dictionnary of the parameters to use for the model
        bounds        (tuple): Bounds to use for the Least Squares

        Returns
        -------
        ccf (array): Crosscorrelation function
        acf (array): Autocorrelation function

        Authors: Allan Denis
        '''

        if len(theta) == 0:
            self._logger.info('No value provided for the list of parameters of the model. Using best vale from the Nested Sampling')
            if not self.fitted:
                raise ForMoSAError("Either run the Nested sampling before computing the CCF or provide a value for the parameters of the model.")# Best params to compute the best model
            theta = self.ns.param_best_dict.copy()
            theta['rv'] = 0
        else:
            theta['rv'] = 0

        self._logger.info(f'Computing rv / vsini map for observation {self.observation.obs_name[indobs]}...')

        adapt = self.config_params['adapt']
        inversion = self.config_params['inversion']

        res_cont = float(adapt['res_cont'][indobs % len(adapt['res_cont'])])
        wav_fit = inversion['wav_fit'][indobs % len(inversion['wav_fit'])]

        lower_bounds_lsq = inversion['hc_lower_bounds_lsq']
        higher_bounds_lsq = inversion['hc_higher_bounds_lsq']
        bounds_lsq = (lower_bounds_lsq[indobs], higher_bounds_lsq[indobs])


        grid_spectro, grid_photo = self.grid.adapted_grid[indobs]['spectro'], self.grid.adapted_grid[indobs]['photo']
        obs_dict_spectro, obs_dict_photo = self.observation.obs_data[indobs]['spectro'], self.observation.obs_data[indobs]['photo']

        res_mod_spectro = grid_spectro.grid.attrs['res']
        res_obs_spectro = obs_dict_spectro['res']

        wav_obs = obs_dict_spectro['wav']
        flx_obs = obs_dict_spectro['flx']
        err_obs = obs_dict_spectro['err']
        star_flx = obs_dict_spectro['star_flx']
        transm = obs_dict_spectro['transm']
        system = obs_dict_spectro['system']

        logL_map = np.zeros((len(vsini_grid), len(rv_grid)))

        # Best nativ model
        theta['vsini'] = 0
        _, mod_dict_spectro = self.ns._compute_model_from_theta(list(theta.values()), obs_dict_spectro, obs_dict_photo, grid_spectro, grid_photo, res_mod_spectro, wav_fit=wav_fit, res_cont=res_cont, bounds_lsq=bounds_lsq)
        vsini_fct = self.ns.params.parameters['vsini'].vsini_function
        wav_mod_spectro = mod_dict_spectro['spectro']['nativ_wav']
        flx_mod_spectro = mod_dict_spectro['spectro']['nativ_flx']   # Using nativ_flx which the flux at the nativ resolution of the model


        for i, vsini_i in enumerate(tqdm(vsini_grid, leave=False)):
            flx_mod_spectro_vsini, res_mod_spectro_vsini = spec.vsini_fct(wav_mod_spectro, flx_mod_spectro, res_mod_spectro, 0.6, vsini_i, vsini_fct)

            # CCF
            logL_map[i] = spec.compute_ccf(wav_mod_spectro, flx_mod_spectro_vsini, wav_obs, flx_obs, err_obs, res_mod_spectro_vsini, res_obs_spectro, res_cont, wav_fit, star_flx, transm, system, rv_grid=rv_grid, rv_sini_map=True, bounds=bounds)

        logL_map -= np.min(logL_map)
        max_indices = np.unravel_index(np.argmax(logL_map), logL_map.shape)  # Indices de la valeur max
        rv_peak, vsini_peak = rv_grid[max_indices[1]], vsini_grid[max_indices[0]]

        # plot
        fig = plt.figure('rv-vsin(i) map', figsize=(10,6))
        plt.clf()
        ax = fig.add_subplot()

        im = ax.pcolormesh(rv_grid, vsini_grid, logL_map, cmap=plt.cm.inferno, rasterized=True)

        ax.set_xlabel('RV [km/s]')
        ax.set_ylabel('$v\\,\\sin i$ [km/s]')

        ax.axhline(y=vsini_peak, linestyle='--', c='C3')
        ax.axvline(x=rv_peak, linestyle='--', c='C3')

        cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('$\\log \\mathcal{L}$', fontsize=22, labelpad=10)

        # add model parameters in the title
        ax.set_title(f'RV / V.sini map\nObs: {self.observation.obs_name[indobs]}\nRV={rv_peak:.1f} km/s, vsini={vsini_peak:.1f} km/s')

        if save:
            filename = self.paths.result_path / f'rv_vsini_map_obs{indobs}.pdf'
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            self._logger.info(f"RV / V.sini map plot saved to {filename}")
        if plot:
            plt.show()

        return fig, ax, logL_map, rv_peak, vsini_peak

