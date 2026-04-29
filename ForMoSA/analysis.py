import logging
import numpy as np

from ForMoSA.config.paths import Paths
from ForMoSA.core.errors import ForMoSAError
from ForMoSA.grid.model_grid import ModelGrid
from ForMoSA.core.loggings import setup_logging
from ForMoSA.grid.subgrid_set import SubGridSet
from ForMoSA.filter.filter import PhotometryFilter
from ForMoSA.nested_sampling.results import NSResults
from ForMoSA.nested_sampling.plotting import Plotting
from ForMoSA.core.config import PLOTS_CONFIG, MAIN_PLOT
from ForMoSA.parameter.parameter_set import ParameterSet
from ForMoSA.nested_sampling.ns_analysis import NSAnalysis
from ForMoSA.grid.subgrid_photometry import SubGridPhotometry
from ForMoSA.observation.observation_set import ObservationSet
from ForMoSA.grid.subgrid_spectroscopy import SubGridSpectroscopy
from ForMoSA.nested_sampling.nested_sampling import NestedSampling
from ForMoSA.core.enums import ObservationType, NestedAlgorithm, LogLikelihoodType
from ForMoSA.config.global_config import ConfigPath, ConfigAdapt, ConfigInversion, ConfigParameters, Config_NS


class Analysis(object):
    '''
    ForMoSA data analysis class.

    Parameters
    ----------
    config_path : ConfigPath
        Instance of class ConfigPath representing the configuration paths.
    adapted : bool
        Whether the model is adapted to the data, by default False. Can be set to True if the model has already been adapted to the data
    fitted : bool
        Whether the data have already been fitted for
    logger : logging.Logger
        Logger
    log_level : str
        Log level of the handler, by default ``'info'`` for all important informations.

    Notes
    -----
    Authors: Allan Denis
    '''

    def __init__(self, config_path: ConfigPath, adapted: bool = False, fitted: bool = False, logger: logging.Logger = None, log_level: str = 'info') -> None:

        self._logger = logger or setup_logging(level=log_level, name='ForMoSA Analysis')

        self._config_path = config_path
        self._adapted = adapted
        self._fitted = fitted
        self._ns = None
        self._parameters = None
        self._ns_analysis = None

        # Paths
        self._paths = Paths(config_path, logger=self._logger)

        # ModelGrid
        self._grid = ModelGrid.from_file(self._paths.model_path, logger=self._logger)

        # Adapted Observations
        # When running with adapted=True, prefer adapted observations saved on disk
        # because they include continuum metadata (wave_cont/res_cont) required by
        # high-contrast modeling.
        if self._adapted:
            try:
                self._observations = ObservationSet.from_npz(self._paths.result_path, logger=self._logger)
                self._logger.info('    Loaded adapted observations from result path')
            except ForMoSAError as e:
                self._logger.warning(f'Recovery of adapted observations from result path {self._paths.result_path} produced the following error: {e}. Trying with the raw FITS observations')
                self._observations = ObservationSet.from_fits(self._paths.observation_path, logger=self._logger)
        else:
            self._observations = ObservationSet.from_fits(self._paths.observation_path, logger=self._logger)

        # Adapted SubGrids
        if self._adapted:
            try:
                self._subgrids = SubGridSet.from_path(self.paths.adapt_store_path, self.grid, logger=self._logger)
            except ForMoSAError as e:
                raise ForMoSAError(f'Recovering SubGridSet from path {self.paths.adapt_store_path} produced the following error: {e}', self.logger)

        # Non adapted SubGrids
        else:
            self._subgrids = SubGridSet(parent_grid=self._grid, logger=self._logger)

        # observations, ns, parameters and ns_analysis
        if self.fitted:
            try:
                self._observations = ObservationSet.from_npz(self._paths.result_path, logger=self._logger)
                self._ns = NestedSampling.from_json(self.paths.result_path, observations=self._observations, subgrids=self._subgrids, logger=self._logger)
                self._parameters = self._ns.parameters
                self._ns_analysis = NSAnalysis(self.ns, logger=self.logger)
            except ForMoSAError as e:
                raise ForMoSAError(f'Recovering NestedSampling from path {self.paths.result_path} produced the following error: {e}. You probably want to fit your data first (set fitted to False)', self._logger)

        # Update configurations for plotting
        for obs in self._observations:
            obs.plot_config.set_plot_config(color=obs.plot_config.cmap(self._observations.mcolors_normalize(obs.central_wavelength)))

        # Propagate the computed colors to restricted_observations.
        # These are deep-copied from self._observations before the loop above,
        # so they keep the default color unless explicitly updated here.
        if self._ns is not None:
            color_by_name = {obs.name: obs.plot_config.color for obs in self._observations}
            for obs in self._ns.restricted_observations:
                if obs.name in color_by_name:
                    obs.plot_config.set_plot_config(color=color_by_name[obs.name])

        # Upade main plot configuration
        MAIN_PLOT.legend_ncol = max(1, (np.sum(
            [obs.nb_filters for obs in self.observations.photometry_observations])
            + np.sum([obs.nb_instruments for obs in self.observations.spectral_observations])
            + 6 - len(self.observations.high_contrast_observations)) // 7)

        MAIN_PLOT.legend_hc_ncol = max(1, (np.sum([obs.nb_instruments for obs in self.observations.high_contrast_observations]) + 6) // 7)

        MAIN_PLOT.legend_filt_ncol = max(1, (np.sum([obs.nb_filters for obs in self.observations.photometry_observations]) + 4) // 5)

    # =======================
    # Properties
    # =======================

    @property
    def adapted(self) -> bool:
        """Whether data and model are adapted."""
        return self._adapted

    @adapted.setter
    def adapted(self, adapted_status: bool) -> bool:
        """Setter for adapted."""
        self._adapted = adapted_status

    @property
    def fitted(self) -> bool:
        """Whether models have been fitted to the data."""
        return self._fitted

    @fitted.setter
    def fitted(self, fitted_status: bool) -> bool:
        self._fitted = fitted_status

    @property
    def logger(self) -> logging.Logger:
        """Logger."""
        return self._logger

    @property
    def config_path(self) -> ConfigPath:
        """ConfigLoader."""
        return self._config_path

    @property
    def observations(self) -> ObservationSet:
        """Set of observations."""
        return self._observations

    @property
    def grid(self) -> ModelGrid:
        """ModelGrid."""
        return self._grid

    @property
    def parameters(self) -> ParameterSet:
        """Set of parameters."""
        return self._parameters

    @property
    def paths(self) -> Paths:
        """ForMoSAPaths."""
        return self._paths

    @property
    def subgrids(self) -> SubGridSet:
        """Set of subgrids."""
        return self._subgrids

    @property
    def ns(self) -> NestedSampling:
        """Nested Sampling."""
        return self._ns

    @property
    def ns_analysis(self) -> NSAnalysis:
        """NSAnalysis."""
        return self._ns_analysis

    # =======================
    # Representation
    # =======================

    def __repr__(self) -> str:
        return 'Analysis()'

    # =======================
    # Methods
    # =======================

    def adapt(self, config_adapt: ConfigAdapt, config_inversion: ConfigInversion, to_json: bool = False) -> None:
        '''
        Adapt the grid of model to each observation.

        Parameters
        ----------
        config_adapt : ConfigAdapt
            Instance of ConfigAdapt

        Notes
        -----
        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''

        # ==================
        # Checks
        # ==================

        # config_adapt type and config_inversion types
        if not isinstance(config_adapt, ConfigAdapt):
            raise ForMoSAError(f'Wrong type for config_adapt: {type(config_adapt)}. Expected a ConfigAdapt', self.logger)

        # Check that lengths of MOSAIC parameters of config_adapt and config_inversion are consistent with number of observations
        try:
            config_adapt._check_with_n_obs(self.observations.n_observations)
            config_inversion._check_with_n_obs(self.observations.n_observations)
        except ForMoSAError as e:
            raise ForMoSAError(e, self.logger)

        # Compute target resolution to reach for the observations
        target_resolution = config_adapt._compute_obs_target_resolution(self.observations, self.grid)
        # Adapt observations
        self.observations.adapt_all(target_resolution = target_resolution, wave_cont = config_adapt.wav_cont, res_cont = config_adapt.res_cont)

        # Save observations
        self.observations.save_all(self.paths.result_path, to_json=to_json)

        if not self.adapted:
            try:
                # Compute target wavelength and resolutions to reach for the subgrids
                target_wave, target_res = target_wave, target_res = config_adapt._compute_model_target_wavelength_and_resolution(self.observations, self.grid)
                # Compute whether to remove continuum for the subgrids
                remove_continuum = config_adapt._determine_remove_continuum(self.observations)

                self._logger.debug(' Generate a set of subgrids from the observations')
                subgrid_set = SubGridSet(self.grid, logger=self.logger)

                # ==================
                # Adapt subgrids
                # ==================

                # Loop in observations
                for obs, wave, res, remove_cont in zip(self.observations.observations, target_wave, target_res, remove_continuum):
                    # Spectroscopic observation
                    if obs.ObsType == ObservationType.SPECTROSCOPIC.obstype:
                        subgrid = SubGridSpectroscopy.from_parent(parent_grid = self.grid, target_wavelength=wave, target_resolution=res, name = obs.name, logger = self.logger, remove_continuum=remove_cont, res_cont=obs._res_cont, wave_cont=obs._wave_cont, backend=config_adapt.backend, n_jobs=config_adapt.n_jobs)
                    # Photometric observation
                    elif obs.ObsType == ObservationType.PHOTOMETRIC.obstype:
                        Filter_list = []
                        for facility, instrument, filter_id in zip(obs.facility, obs.instrument, obs.filter_id):
                            Filter_list.append(PhotometryFilter(facility, instrument, filter_id))
                        subgrid = SubGridPhotometry.from_parent(parent_grid = self.grid, Filter = Filter_list, name = obs.name, logger = self.logger, backend=config_adapt.backend, n_jobs=config_adapt.n_jobs)
                    # Unknown type
                    else:
                        raise ForMoSAError(f'Unknown ObservationType: {obs.ObsType}')

                    subgrid_set.add_subgrid(subgrid)

            except ForMoSAError as e:
                raise ForMoSAError(e, self.logger)

            self._logger.info(f'    Set of subgrids generated: {subgrid_set.subgrid_names}')
            self._subgrids = subgrid_set

            # Interpolate mmissing values in the subgrids
            self.subgrids.interpolate_all(config_adapt.method)

            # Save all the subgrids
            self.subgrids.save_all(self.paths.adapt_store_path)

            # Set adapted to True
            self._adapted = True

    def nested_sampling(self, config_parameters: ConfigParameters, config_adapt: ConfigAdapt = ConfigAdapt(), config_inversion: ConfigInversion = ConfigInversion(), config_NS: Config_NS = Config_NS()) -> None:
        '''
        Launch nested sampling.

        Parameters
        ----------
        config_adapt : ConfigAdapt
            Instance of class ConfigAdapt
        config_inversion : ConfigInversion
            Instance of class ConfigInversion
        config_parameters : ConfigParameters
            Instance of class ConfigParameters

        Notes
        -----
        Authors: Allan Denis
        '''

        for config, config_type in zip([config_parameters, config_adapt, config_inversion, config_NS], [ConfigParameters, ConfigAdapt, ConfigInversion, Config_NS]):
            if not isinstance(config, config_type):
                raise ForMoSAError(f'Wrong type for {config} : {type(config)}. Expected a {config_type}')

        config_adapt._check_with_n_obs(self.observations.n_observations)
        config_inversion._check_with_n_obs(self.observations.n_observations)

        # Propagate the configurations to restricted_observations.
        # In case they have been changed in the observations, they need to be deep copied in the restricted observations
        if self._ns is not None:
            config_by_name = {obs.name: obs._plot_config.to_dict() for obs in self._observations}

            for obs in self._ns.restricted_observations.observations:
                if obs.name in config_by_name:
                    obs._plot_config.set_plot_config(**config_by_name[obs.name])

        if not self.fitted:

            # ==================
            # Checks
            # ==================

            # Build set of parameters
            self._parameters = ParameterSet.from_config(config_parameters, logger=self.logger)

            # Replace 'parX' names by associated physical parameters ('Teff', 'logg', ...)
            for i, name in enumerate(self.parameters.names):
                if name.startswith('par'):  # Detect grid parameters
                    self.parameters.parameters[i]._title = self.grid.titles[self.grid.keys.index(name)]  # Rename parameter with title associated to 'parX'

            algorithm, npoints, logL_type_list = config_inversion.ns_algo, config_inversion.npoints, config_inversion.logL_type
            algorithm = NestedAlgorithm[algorithm.upper()]
            logL_type = [LogLikelihoodType[logL.upper()] for logL in logL_type_list]

            # Create instance of NestedSampling
            self._ns = NestedSampling(
                algorithm=algorithm,
                npoints=npoints,
                logL_type=logL_type,
                config_NS= config_NS,
                observations=self.observations,
                subgrids=self.subgrids,
                parameters=self.parameters,
                wave_fit=config_inversion.wav_fit,
                interp_method=config_adapt.method,
                bounds_lsq=config_inversion.hc_bounds
                )

            # Launch NestedSampling
            self._ns.run(results_path=self.paths.result_path)

            # Create instance of NSAnalysis
            self._ns_analysis = NSAnalysis(self._ns, logger=self.logger)

            # Save results
            self._ns.save_results(self.paths.result_path)

            # Set fitted to True
            self._fitted = True

    def plot(self, results: NSResults, save: bool = True, plot_native_model: bool = False) -> None:
        '''
        Plot the results.

        Parameters
        ----------
        results : NSResults
            An instance of NSResults
        save : bool
            Whether to save the results
        plot_native_model : bool
            Whether to plot the native model

        Notes
        -----
        Authors: Allan Denis
        '''

        # Initial checks
        if not isinstance(results, NSResults):
            raise ForMoSAError(f'Wrong type for results: {type(results)}. Expected NSResults')

        self.plots = Plotting(results, self.logger)

        fig_corner = self.plots.plot_corner()

        if save:
            path = self.paths.result_path / 'corner.pdf'
            fig_corner.savefig(path, dpi=300, bbox_inches='tight')

        fig_chains, axs = self.plots.plot_chains()

        if save:
            path = self.paths.result_path / 'chains.pdf'
            fig_chains.savefig(path, dpi=300, bbox_inches='tight')

        fig_radar, ax = self.plots.plot_radars()

        if save:
            path = self.paths.result_path / 'radar.pdf'
            fig_radar.savefig(path, dpi=300, bbox_inches='tight')

        # Get native best fit from ns_analysis if plot_native_model is True, otherwise set it to None to avoid unnecessary computations in the plot_fit function
        native_best_fit = None
        if plot_native_model:
            native_best_fit = self.ns_analysis.native_best_fit

        # Plot best fit for each observation, with the native model if requested
        fig_best_fit, ax, ax_filt, axr, axr2 = self.plots.plot_fit(self.ns.restricted_observations, self.ns_analysis.best_fit, plot_native_model=plot_native_model, native_model=native_best_fit)

        # If requested, plot the 1-sigma and 2-sigma confidence intervals of the best fit in the best fit plot
        if plot_native_model:
            lower_1_sigma, higher_1_sigma = self.ns_analysis.best_fit_interval(perc=0.68)
            lower_2_sigma, higher_2_sigma = self.ns_analysis.best_fit_interval(perc=0.95)

            ax.fill_between(lower_1_sigma.wave, lower_1_sigma.flux, higher_1_sigma.flux, color='grey', alpha=0.5, zorder=PLOTS_CONFIG.BestFitPlot.zorder, label='1-sig interval')
            ax.fill_between(lower_2_sigma.wave, lower_2_sigma.flux, higher_2_sigma.flux, color='grey', alpha=0.2, zorder=PLOTS_CONFIG.BestFitPlot.zorder, label='2-sig interval')

        if save:
            path = self.paths.result_path / 'best_fit.pdf'
            fig_best_fit.savefig(path, dpi=300, bbox_inches='tight')

    # =========================================
    # CCF Plotting Functions
    # =========================================
    def plot_ccf(self, rv_grid: np.ndarray, save_fig: bool = True, save_results: bool = False) -> None:
        '''
        Compute and optionally plot the Cross-Correlation Function (CCF).

        Parameters
        ----------
        rv_grid : np.ndarray
            Grid of radial velocity values (in km/s)
        save_fig : bool
            Whether to save the figure
        save_results : bool
            Whether to save the results of the CCF computation

        Notes
        -----
        Authors: Bhavesh Rajpoot (adapted from Allan Denis)
        '''

        if self.ns is None or self.ns.results is None:
            raise ForMoSAError('Please first run the Nested Sampling before computing the CCF', self.logger)

        if not hasattr(self, 'plots') or self.plots is None:
            self.plots = Plotting(self.ns.results, self.logger)

        # initialize save_path based on save_results or save_fig
        save_path = self.paths.result_path if (save_results or save_fig) else None

        for index in range(self.observations.n_observations):
            ccf_dict = self.ns_analysis.compute_ccf(rv_grid, index=index)
            file_tag = list(ccf_dict.keys())[0]
            rv_grid, ccf, acf, ccf_star, _, _ = list(ccf_dict[file_tag].values())
            fig, ax = self.plots.plot_ccf(rv_grid, ccf, acf, ccf_star=ccf_star, title=file_tag)

            if save_path:
                path = self.paths.result_path / f'ccf_{file_tag}.pdf'
                fig.savefig(path)

            if save_results:
                results_path = self.paths.result_path / f'ccf_results_{file_tag}.npz'

                # save the ccf_dict to a .npz file
                np.savez(results_path, **ccf_dict[file_tag])


    def plot_rv_vsini_map(self, rv_grid: np.ndarray, vsini_grid: np.ndarray, save_fig: bool = True, save_results: bool = False) -> None:
        '''
        Compute and optionally plot the RV vs v.sin(i) loglikelihood map.

        Parameters
        ----------
        rv_grid : np.ndarray
            Grid of radial velocity values (in km/s)
        vsini_grid : np.ndarray
            Grid of v.sin(i) values (in km/s)
        save_fig : bool
            Whether to save the figure
        save_results : bool
            Whether to save the results of the RV-vsini map computation

        Notes
        -----
        Authors: Bhavesh Rajpoot (adapted from Allan Denis)
        '''

        if self.ns is None or self.ns.results is None:
            raise ForMoSAError('Please first run the Nested Sampling before computing the RV-vsini map', self.logger)

        if not hasattr(self, 'plots') or self.plots is None:
            self.plots = Plotting(self.ns.results, self.logger)

        save_path = self.paths.result_path if (save_fig or save_results) else None

        for index in range(self.observations.n_observations):
            rv_vsini_map = self.ns_analysis.compute_rv_vsini_map(rv_grid, vsini_grid, index=index)
            file_tag = list(rv_vsini_map.keys())[0]
            rv_grid, vsini_grid, logL_map, _, _ = tuple(rv_vsini_map[file_tag].values())
            fig, ax = self.plots.plot_rv_vsini_map(rv_grid, vsini_grid, logL_map, title=file_tag)

            if save_path:
                path = self.paths.result_path / f'rv_vsini_map_{file_tag}.pdf'
                fig.savefig(path, dpi=300, bbox_inches='tight')

            if save_results:
                results_path = self.paths.result_path / f'rv_vsini_map_results_{file_tag}.npz'

                # save the rv_vsini_map to a .npz file
                np.savez(results_path, **rv_vsini_map[file_tag])