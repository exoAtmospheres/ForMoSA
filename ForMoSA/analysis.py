import logging

from ForMoSA.core.config import PLOTS_CONFIG
from ForMoSA.core.errors import ForMoSAError
from ForMoSA.config.paths import ForMoSAPaths
from ForMoSA.grid.model_grid import ModelGrid
from ForMoSA.core.loggings import setup_logging
from ForMoSA.grid.subgrid_set import SubGridSet
from ForMoSA.filter.filter import PhotometryFilter
from ForMoSA.nested_sampling.results import NSResults
from ForMoSA.nested_sampling.plotting import Plotting
from ForMoSA.parameter.parameter_set import ParameterSet
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
    config                (ConfigLoader): Instance of class ConfigLoader representing the configuration parameters.
    adapted                       (bool): Whether the model is adapted to the data, by default False. Can be set to True if the model has already been adapted to the data
    fitted                        (bool): Whether the data have already been fitted for
    log_level (str): og level of the handler, by default ``'info'`` for all important informations.

    Authors: Allan Denis
    '''

    def __init__(self, config_path: ConfigPath, adapted: bool = False, fitted: bool = False, log_level: str = 'info') -> None:

        self._logger = setup_logging(level=log_level, name='ForMoSA Analysis')

        self._config_path = config_path
        self._adapted = adapted
        self._fitted = fitted
        self._ns = None
        self._parameters = None

        # Paths
        self._paths = ForMoSAPaths(config_path)

        # ModelGrid
        self._grid = ModelGrid.from_file(self._paths.model_path)

        # Adapt Observations
        self._observations = ObservationSet.from_fits(self._paths.observation_path, logger=self._logger)

        # Adapted SubGrids
        if self._adapted:
            self._subgrids = SubGridSet.from_path(self.paths.adapt_store_path, self.grid, logger=self._logger)
        # Non adapted SubGrids
        else:
            # SubGrids
            self._subgrids = SubGridSet(parent_grid=self._grid, logger=self._logger)

        if self._fitted:
            try:
                self._ns = NestedSampling.from_json(self._paths.result_path, observations=self._observations, subgrids=self._subgrids, logger=self._logger)
                self._parameters = self._ns.parameters
            except ForMoSAError as e:
                raise ForMoSAError(f'Recovering NestedSampling produced the following error: {e}. You probably want to fit your data first (set fitted to False)', self._logger)

    # =======================
    # Properties
    # =======================

    @property
    def adapted(self) -> bool:                                      # Whether data and model are adapted
        return self._adapted

    @adapted.setter
    def adapted(self, adapted_status: bool) -> bool:                # Setter for adapted
        self._adapted = adapted_status

    @property
    def fitted(self) -> bool:                                       # Whether models have been fitted to the data
        return self._fitted

    @fitted.setter
    def fitted(self, fitted_status: bool) -> bool:
        self._fitted = fitted_status

    @property
    def logger(self) -> logging.Logger:                             # Logger
        return self._logger

    @property
    def config_path(self) -> ConfigPath:                            # ConfigLoader
        return self._config_path

    @property
    def observations(self) -> ObservationSet:                       # Set of observations
        return self._observations

    @property
    def grid(self) -> ModelGrid:                                    # ModelGrid
        return self._grid

    @property
    def parameters(self) -> ParameterSet:                           # Set of parameters
        return self._parameters

    @property
    def paths(self) -> ForMoSAPaths:                                # ForMoSAPaths
        return self._paths

    @property
    def subgrids(self) -> SubGridSet:                               # Set of subgrids
        return self._subgrids

    @property
    def ns(self) -> NestedSampling:                                 # Nested Sampling
        return self._ns

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
        config_adapt (ConfigAdapt): Instance of ConfigAdapt

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
        self.observations.adapt_all(target_resolution = target_resolution, wave_cont = config_inversion.wav_fit, res_cont = config_adapt.res_cont)

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
                for obs, wave, res, remove_cont, res_cont, wave_cont in zip(self.observations.observations, target_wave, target_res, remove_continuum, config_adapt.res_cont, config_inversion.wav_fit):
                    # Spectroscopic observation
                    if obs.ObsType == ObservationType.SPECTROSCOPIC.obstype:
                        subgrid = SubGridSpectroscopy.from_parent(parent_grid = self.grid, target_wavelength=wave, target_resolution=res, name = obs.name, logger = self.logger, remove_continuum=remove_cont, res_cont=res_cont, wave_cont=wave_cont)
                    # Photometric observation
                    elif obs.ObsType == ObservationType.PHOTOMETRIC.obstype:
                        Filter = PhotometryFilter(obs.facility, obs.instrument, obs.filter_id)
                        subgrid = SubGridPhotometry.from_parent(parent_grid = self.grid, Filter = Filter, name = obs.name, logger = self.logger)
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
        config_adapt           (ConfigAdapt): Instance of class ConfigAdapt
        config_inversion   (ConfigInversion): Instance of class ConfigInversion
        config_parameters (ConfigParameters): Instance of class ConfigParameters

        Authors: Allan Denis
        '''

        for config, config_type in zip([config_parameters, config_adapt, config_inversion, config_NS], [ConfigParameters, ConfigAdapt, ConfigInversion, Config_NS]):
            if not isinstance(config, config_type):
                raise ForMoSAError(f'Wrong type for {config} : {type(config)}. Expected a {config_type}')

        config_adapt._check_with_n_obs(self.observations.n_observations)
        config_inversion._check_with_n_obs(self.observations.n_observations)

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

            # Save results
            self._ns.save_results(self.paths.result_path)

            # Set fitted to True
            self._fitted = True

    def plot(self, results: NSResults, save: bool = True, plot_native_model: bool = False) -> None:
        '''
        Plot the results.

        Parameters
        ----------
        results (NSResults): An instance of NSResults
        save                (bool): Whether to save the results
        plot_native_model   (bool): Whether to plot the native model

        Authors: Allan Denis
        '''

        # Initial checks
        if not isinstance(results, NSResults):
            raise ForMoSAError(f'Wrong type for results: {type(results)}. Expected NSResults')

        self.plots = Plotting(results, self.logger)

        fig_corner = self.plots.plot_corner()

        if save:
            path = self.paths.result_path / 'corner.pdf'
            fig_corner.savefig(path)

        fig_chains, axs = self.plots.plot_chains()

        if save:
            path = self.paths.result_path / 'chains.pdf'
            fig_chains.savefig(path)

        fig_radar, ax = self.plots.plot_radars()

        if save:
            path = self.paths.result_path / 'radar.pdf'
            fig_radar.savefig(path)

        native_best_fit = None
        if plot_native_model:
            native_best_fit = self.ns.native_best_fit

        fig_best_fit, ax, ax_filt, axr, axr2 = self.plots.plot_fit(self.observations, self.ns.best_fit, plot_native_model=plot_native_model, native_model=native_best_fit)
        if plot_native_model:
            lower_1_sigma, higher_1_sigma = self.ns.best_fit_interval(perc=0.68)
            lower_2_sigma, higher_2_sigma = self.ns.best_fit_interval(perc=0.95)

            ax.fill_between(lower_1_sigma.wave, lower_1_sigma.flux, higher_1_sigma.flux, color='grey', alpha=0.5, zorder=PLOTS_CONFIG.BestFitPlot.zorder)
            ax.fill_between(lower_2_sigma.wave, lower_2_sigma.flux, higher_2_sigma.flux, color='grey', alpha=0.2, zorder=PLOTS_CONFIG.BestFitPlot.zorder)


        if save:
            path = self.paths.result_path / 'best_fit.pdf'
            fig_best_fit.savefig(path)
