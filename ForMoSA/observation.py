import numpy as np
import os
import glob

from pathlib import Path
from scipy.interpolate import interp1d
from astropy.io import fits

from ForMoSA.utils.spec import resolution_decreasing, continuum_estimate
from ForMoSA.error import ForMoSAError

class Observation(object):
    '''
    ForMoSA observation class, which provides easy access to an observation

    Parameters
    ----------
    observation_path (str | os.PathLike): Path to the observation file(s) (multiple observation files can exist)
    logger                      (logger): Logger
    config_plotting               (dict): Dictionary of the plotting configuration {'color': color, 'edgecolors': edgecolor, 'marker': marker, 'size': size}

    Authors: Allan Denis
    '''

    def __init__(self, observation_path: str | os.PathLike, logger, config_plotting: dict = dict()) -> None:
        if str(observation_path).endswith('.fits'):
            self._observation_path = Path(str(observation_path)).expanduser()
        else:
            self._observation_path = Path(str(observation_path).rstrip('*') + '/*').expanduser()
        self._obs_data = dict()
        self._logger = logger
        self._config_plotting = config_plotting
        self._instrument_files = dict()

        # extract observation data
        self._extract_observation()

    ##################################################
    # Representation
    ##################################################

    def __repr__(self) -> str:
        return f'<Observation : {self.obs_name}>'

    def __format__(self) -> str:
        return self.__repr__()


    ##################################################
    # Properties
    ##################################################


    @property
    def observation_path(self):
        return self._observation_path

    @property
    def root(self):
        return Path(str(self._observation_path).rstrip('*'))

    @property
    def obs_files(self):
        files = [f for f in glob.glob(str(self.observation_path)) if f.lower().endswith('.fits')]
        files = sorted(files)
        if len(files) == 0:  # No observation
            msg = f' No observation. {self.observation_path} does not contain any observation.'
            self._logger.error(msg)
            return ForMoSAError(msg)
        else:
            return {i: file for i, file in enumerate(files)}

    @property
    def obs_files_list(self):
        return [self.obs_files[key] for key in range(self.n_files)]

    @property
    def obs_name(self):
        _obs_name = dict()
        for i, file in enumerate(self.obs_files.values()):
            for instrument in self.instrument_files[i]:
                elements = (file.split('/')[-1].split('.fits')[0] + f'_{instrument}').split('_')
                seen = set()
                unique_elements = [x for x in elements if not (x in seen or seen.add(x))]
                _obs_name[len(_obs_name)] = '_'.join(unique_elements)
        return _obs_name

    @property
    def obs_name_list(self):
        return [self.obs_name[key] for key in range(self.n_obs)]

    @property
    def obs_data(self):
        return self._obs_data

    @property
    def n_files(self):
        return len(self.obs_files)

    @property
    def n_obs(self):
        return(len(self.obs_data))

    @property
    def max_resolution(self):
        resolutions = [res for i in range(self.n_obs) for res in self.obs_data[i]['spectro']['res']]

        return max(resolutions)

    @property
    def wave_from_max_resolution(self):
        return

    @property
    def instruments(self):
        return {i: ins
                for i in self.obs_data.keys()
                for comp_type in ('spectro', 'photo')
                for ins in self.obs_data[i][comp_type]['ins']}

    @property
    def instrument_files(self):
        if not (self._instrument_files):
            msg = ' You need to extract the data first.'
            self._logger.error(msg)
            raise ForMoSAError(msg)

        return self._instrument_files

    @property
    def config_plotting(self):
        _config_plotting = dict()
        for indobs in range(self.n_obs):
            _config_plotting[indobs] = {'color': self._config_plotting['color'][indobs],
                                        'edgecolor': self._config_plotting['edgecolor'][indobs],
                                        'marker': self._config_plotting['marker'][indobs],
                                        'size': float(self._config_plotting['size'][indobs])}
        return _config_plotting


    ##################################################
    # Methods
    ##################################################

    def _extract_observation(self) -> dict:
        """
        Method to extract the information from the observation files
        Extracts the wavelengths (um - vacuum), flux (W.m-2.um-1), errors (W.m-2.um-1), covariance (W.m-2.um-1)**2,
        spectral resolution, instrument/filter name, transmission (Atmo+inst) and star flux (W.m-2.um-1).
        Sort the data by instrument name and returns a dictionary containing one spectroscopic and one photometric observation for each instrument.

        Returns:
            obs_dict (dict): Dictionaries containing all the extracted data, separated by instrument.

        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        """

        obs_dict = dict()
        instrument_files = dict()

        indobs = 0

        for indfile in range(self.n_files):
            instrument_files[indfile] = []
            with fits.open(self.obs_files[indfile]) as hdul:
                self._logger.debug(f'> Read file {self.obs_files[indfile]}.')

                missing_keys = []

                # Check that parameters are well defined
                try:
                    wav = hdul[1].data['WAV']
                except KeyError:
                    missing_keys.append('WAV')
                try:
                    flx = hdul[1].data['FLX']
                except KeyError:
                    missing_keys.append('FLX')
                try:
                    res = hdul[1].data['RES']
                except KeyError:
                    missing_keys.append('RES')
                try:
                    ins = np.asarray(hdul[1].data['INS'], dtype=str)
                except KeyError:
                    missing_keys.append('INS')

                if len(missing_keys) > 0:
                    msg = f"Keys missing: {', '.join(missing_keys)}"
                    self._logger.critical(msg)
                    raise ForMoSAError(msg)

                # Errors and covariances
                try:
                    err = hdul[1].data['ERR']
                    cov = np.asarray([])
                    self._logger.info(f'Your observation {self.obs_files[indfile]} contains an error vector.')
                except KeyError:
                    missing_keys.append('ERR')
                    try:
                        cov = hdul[1].data['COV']
                        err = np.sqrt(np.diag(np.abs(cov)))
                        self._logger.info(f'Your observation {self.obs_files[indfile]} contains a covariance matrix.')
                    except KeyError:
                        missing_keys.append('COV')
                        msg = f"One of the following keys is missing: {', '.join(missing_keys)}"
                        self._logger.critical(msg)
                        raise ForMoSAError(msg)

                # High-contrast parameters
                try:
                    transm = hdul[1].data['TRANSM']
                    self._logger.info(f'Your observation {self.obs_files[indfile]} contains a transmission vector.')
                except:
                    transm = np.array([])
                try:
                    star_flx = hdul[1].data['STAR_FLX1'][:, np.newaxis]
                    is_star = True
                except:
                    star_flx = np.array([])
                    is_star = False
                if is_star:
                    i = 2
                    while True:
                        try:
                            star_flx = np.concatenate((star_flx, hdul[1].data['STAR_FLX' + str(i)][:, np.newaxis]), axis=1)
                            i += 1
                        except:
                            self._logger.info(f'Your observation {self.obs_files[indfile]} contains {i-1} star vectors.')
                            break

                # Additional systematics parameters
                try:
                    is_system = True
                    system = hdul[1].data['SYSTEMATICS1'][:, np.newaxis]
                except:
                    is_system = False
                    system = np.array([])
                if is_system:
                    i = 2
                    while True:
                        try:
                            system = np.concatenate((system, hdul[1].data['SYSTEMATICS' + str(i)][:, np.newaxis]), axis=1)
                            i += 1
                        except:
                            self._logger.info(f'Your observation {self.obs_files[indfile]} contains {i-1} systematics vectors.')
                            break

                # Filter the NaN and inf values
                self._logger.debug('> Detect NaN data.')
                nan_mod_ind = (~np.isnan(flx)) & (~np.isnan(err)) & (np.isfinite(flx)) & (np.isfinite(err))
                if cov.size != 0:
                    nan_mod_ind = (nan_mod_ind) & np.all(~np.isnan(cov), axis=0) & np.all(~np.isnan(cov), axis=1) & np.all(np.isfinite(cov), axis=0) & np.all(np.isfinite(cov), axis=1)
                if transm.size != 0:
                    nan_mod_ind = (nan_mod_ind) & (~np.isnan(transm)) & (np.isfinite(transm))
                if star_flx.size != 0:
                    for i in range(star_flx.shape[1]):
                        nan_mod_ind = (nan_mod_ind) & (~np.isnan(star_flx.T[i])) & (np.isfinite(star_flx.T[i]))
                if system.size != 0:
                    for i in range(system.shape[1]):
                        nan_mod_ind = (nan_mod_ind) & (~np.isnan(system.T[i])) & (np.isfinite(system.T[i]))
                self._logger.info(f'> {len(nan_mod_ind[nan_mod_ind == False])} NaN data out of {len(flx)}.')

                # Apply the NaN filter to all arrays
                wav = wav[nan_mod_ind]
                flx = flx[nan_mod_ind]
                res = res[nan_mod_ind]
                ins = ins[nan_mod_ind]
                err = err[nan_mod_ind]
                if transm.size != 0:
                    transm = transm[nan_mod_ind]
                if star_flx.size != 0:
                    star_flx = star_flx[nan_mod_ind]
                if system.size != 0:
                    system = system[nan_mod_ind]

                # Get unique instruments
                unique_ins = np.unique(ins)

                # Initialize obs_dict with instrument keys
                for inst in unique_ins:
                    elements = (self.obs_files[indfile].split('/')[-1].split('.fits')[0] + f'_{inst}').split('_')
                    seen = set()
                    unique_elements = [x for x in elements if not (x in seen or seen.add(x))]
                    obs_name = '_'.join(unique_elements)

                    instrument_files[indfile].append(inst)
                    obs_dict[indobs] = {
                        'spectro': {
                            'wav': np.array([], dtype=float),
                            'flx': np.array([], dtype=float),
                            'err': np.array([], dtype=float),
                            'res': np.array([], dtype=float),
                            'ins': np.array([], dtype=str),
                            'inv_cov': np.array([]),
                            'transm': np.array([]),
                            'star_flx': np.array([]),
                            'speckles': np.array([]),
                            'system': np.array([]),
                            'name': ''
                        },
                        'photo': {
                            'wav': np.array([], dtype=float),
                            'flx': np.array([], dtype=float),
                            'err': np.array([], dtype=float),
                            'ins': np.array([], dtype=str),
                            'name': ''
                        }
                    }

                    # Populate obs_dict by instrument
                    # Mask for this instrument
                    mask_inst = (ins == inst)

                    # Filter global data for this instrument
                    wav_inst = wav[mask_inst]
                    flx_inst = flx[mask_inst]
                    err_inst = err[mask_inst]
                    res_inst = res[mask_inst]
                    ins_inst = ins[mask_inst]
                    if cov.size != 0:
                        cov_inst = cov[mask_inst][:, mask_inst]
                        inv_cov_inst = np.linalg.inv(cov_inst) if cov_inst.size > 0 else np.array([])
                    else:
                        inv_cov_inst = np.array([])
                    transm_inst = transm[mask_inst] if transm.size > 0 else np.array([])
                    star_flx_inst = star_flx[mask_inst] if star_flx.size > 0 else np.array([])
                    system_inst = system[mask_inst] if system.size > 0 else np.array([])

                    # Separate spectroscopic and photometric data for this instrument
                    mask_photo_inst = (res_inst == 0.0)
                    mask_spectro_inst = ~mask_photo_inst

                    # Spectroscopic data
                    obs_dict[indobs]['spectro']['wav'] = wav_inst[mask_spectro_inst]
                    obs_dict[indobs]['spectro']['flx'] = flx_inst[mask_spectro_inst]
                    obs_dict[indobs]['spectro']['err'] = err_inst[mask_spectro_inst]
                    obs_dict[indobs]['spectro']['res'] = res_inst[mask_spectro_inst]
                    obs_dict[indobs]['spectro']['ins'] = np.unique(ins_inst[mask_spectro_inst])
                    if inv_cov_inst.size > 0 and len(wav_inst[mask_spectro_inst]) > 0:
                        obs_dict[indobs]['spectro']['inv_cov'] = inv_cov_inst
                    if transm_inst.size > 0 and len(wav_inst[mask_spectro_inst]) > 0:
                        obs_dict[indobs]['spectro']['transm'] = transm_inst[mask_spectro_inst]
                    if star_flx_inst.size > 0 and len(wav_inst[mask_spectro_inst]) > 0:
                        obs_dict[indobs]['spectro']['star_flx'] = star_flx_inst[mask_spectro_inst]
                    if system_inst.size > 0 and len(wav_inst[mask_spectro_inst]) > 0:
                        obs_dict[indobs]['spectro']['system'] = system_inst[mask_spectro_inst]
                    if len(wav_inst[mask_spectro_inst]) > 0:
                        obs_dict[indobs]['spectro']['name'] = obs_name

                    # Photometric data
                    obs_dict[indobs]['photo']['wav'] = wav_inst[mask_photo_inst]
                    obs_dict[indobs]['photo']['flx'] = flx_inst[mask_photo_inst]
                    obs_dict[indobs]['photo']['err'] = err_inst[mask_photo_inst]
                    obs_dict[indobs]['photo']['ins'] = np.unique(ins_inst[mask_photo_inst])
                    if len(wav_inst[mask_photo_inst]) > 0:
                        obs_dict[indobs]['photo']['name'] = obs_name

                    obs_dict[indobs]['transmission'] = dict()
                    obs_dict[indobs]['obs_name'] = obs_name

                    indobs += 1

        self._obs_data = obs_dict
        self._instrument_files = instrument_files


    def adapt_all_observations(self, res_obs: list, model_wavelength: np.ndarray, model_resolution: np.ndarray, res_cont: list = ['NA'], wav_cont: list = ['NA']):
        ''''
        Decrease the spectral resolution of the current observation and remove the continuum if necessary

        Args:
            res_obs                  (list): Target resolution of each of the observation (['obs', ...] or [float, ...])
            model_wavelength   (np.ndarray): Wavelength grid of the grid of models
            model_resolution   (np.ndarray): Resolution of the grid of models
            res_cont                 (list): Resolution of the continuum of each of the observation ([float | 'NA', ...])
            wav_cont                 (list): Wavelength of the continuum of each of the observation ([np.ndarray | 'NA', ...])

        Returns:
            obs_data          (dict): Adapted current observation

        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''

        for indobs in range(self.n_obs):
            self._logger.info(f" Current observation : {self.obs_data[indobs]['obs_name']}")
            # Decrease the resolution and remove the continuum if necessary
            obs_data = self.obs_data[indobs]['spectro']

            # Determine target resolution to reach for the observation
            target_res_obs = self._set_obs_target_resolution(res_obs[indobs % len(res_obs)], obs_data['wav'], obs_data['res'], model_wavelength, model_resolution)

            # Determine continuum types
            self._obs_data[indobs]['spectro'] = self._adapt_observation(obs_data, target_res_obs, model_wavelength, model_resolution, res_cont = res_cont[indobs % len(res_cont)], wav_cont = wav_cont[indobs % len(wav_cont)])


    def _adapt_observation(self, obs_data: dict, target_res_obs: np.ndarray, model_wavelength: np.ndarray, model_resolution: np.ndarray, res_cont: str = 'NA', wav_cont: str | np.ndarray = 'NA'):
        '''
        Decrease the spectral resolution of the current observation and remove the continuum if necessary

        Args:
            obs_data                 (dict): Dictionary of observation data {'wav': wav, 'flx': flx, ...}
            target_res_obs     (np.ndarray): Target resolution of each of the observation
            model_wavelength   (np.ndarray): Wavelength grid of the grid of models
            model_resolution   (np.ndarray): Resolution of the grid of models
            res_cont          (str | float): Resolution of the continuum of each of the observation
            wav_cont          (str | float): Wavelength of the continuum of each of the observation

        Returns:
            obs_data          (dict): Adapted current observation

        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''

        obs_name = obs_data['name']

        # If we want to decrease the resolution of the spectroscopic data:
        self._logger.info(f' Target resolution for the data: {target_res_obs}.')
        self._logger.debug('> Decrease the resolution of the flux if necessary.')
        obs_data['flx'] = resolution_decreasing(obs_data['wav'],
                                                obs_data['flx'],
                                                obs_data['res'],
                                                obs_data['wav'],
                                                target_res_obs)

        if len(obs_data['transm']) > 0:
            self._logger.debug('> Decrease the resolution of the transmission if necessary..')
            obs_data['transm'] = resolution_decreasing(obs_data['wav'],
                                                       obs_data['transm'],
                                                       obs_data['res'],
                                                       obs_data['wav'],
                                                       target_res_obs)
        if len(obs_data['star_flx']) > 0:
            self._logger.debug('> Decrease the resolution of the star flux if necessary.')
            obs_data['star_flx'] = np.asarray([resolution_decreasing(obs_data['wav'],
                                                                     obs_data['star_flx'][:,i],
                                                                     obs_data['res'],
                                                                     obs_data['wav'],
                                                                     target_res_obs) for i in range(obs_data['star_flx'].shape[-1])]).T

        if len(obs_data['system']) > 0:
            self._logger.debug('> Decrease the resolution of the systematics if necessary.')
            obs_data['system'] = np.asarray([resolution_decreasing(obs_data['wav'],
                                                                   obs_data['system'][:,i],
                                                                   obs_data['res'],
                                                                   obs_data['wav'],
                                                                   target_res_obs) for i in range(obs_data['system'].shape[-1])]).T


        # Since the resolution of the observation might have change, we need to save the new one
        obs_data['res_spectro'] = target_res_obs

        # If we want to estimate and substract the continuum of the data:
        if res_cont != 'NA':
            res_cont = float(res_cont)
            self._logger.debug('> Substract the continuum to the data')

            obs_data['flx_cont'] = continuum_estimate(obs_data['wav'],
                                                      obs_data['flx'],
                                                      obs_data['res'],
                                                      wav_cont, res_cont)

            # High-contrast mode, we do not remove the continuum to the data yet
            if len(obs_data['star_flx']) > 0:
                self._logger.debug('> Estimate the continuum to the stellar data')
                obs_data['star_flx_cont'] = continuum_estimate(obs_data['wav'],
                                                               obs_data['star_flx'][:,len(obs_data['star_flx'][0]) // 2], # Continuum of the star on the central pixel
                                                               obs_data['res'],
                                                               wav_cont, res_cont)


            # Non high-contrast mode, we remove the continuum to the data
            else:
                obs_data['flx'] -= obs_data['flx_cont']

        self._logger.info(f' Observation {obs_name} adapted.')

        return obs_data


    def _set_obs_target_resolution(self, target_res_obs : str, wav_obs_spectro: np.ndarray, res_obs_spectro: np.ndarray, grid_wavelength: np.ndarray, grid_resolution: np.ndarray) -> np.ndarray:
        '''
        Method to set the target resolution of the observation

        Parameters
        ----------
        target_res_obs (str | float): Target resolution of the observation ('obs' or float)
        res_mod_obs     (np.ndarray): Wavelength grid of the observation
        res_obs_spectro (np.ndarray): Resolution of the observation

        Returns:
            target_res_obs (np.ndarray): Target resolution of the observation

        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''

        # Setup target resolution for the observation
        # Interpolate the resolution of the model onto the wavelength of the data to properly decrease the resolution if necessary
        interp_mod_to_obs = interp1d(grid_wavelength, grid_resolution, fill_value='extrapolate')
        res_mod_obs = interp_mod_to_obs(wav_obs_spectro)

        if target_res_obs == 'obs': # Keeping the resolution of the observation except where its higher than the model's
            target_res_obs = np.minimum(res_obs_spectro, res_mod_obs)
        else:                                             # Using a custom resolution except where its higher than the model's or the observation's
            res_custom = np.full(len(res_obs_spectro), float(target_res_obs))
            target_res_obs = np.minimum(res_obs_spectro, res_mod_obs, res_custom)

        return target_res_obs


    def _save_observation_file(self, path: str | os.PathLike, indobs: int = 0) -> None:
        '''
        Method to save the observation indobs

        Parameters
        ----------
        path    (str | os.PathLike): Path to save the date to
        indobs                (int): Index of the observation to save

        Authors: Allan Denis
        '''

        if not(os.path.isdir(path)):
            msg = f' {path} does not exist'
            self._logger.error(msg)
            raise ForMoSAError(msg)

        self._logger.info(f' Save observation {self.obs_name[indobs]}')
        self._logger.debug(f'> Save observation file to {path}' + f'/spectrum_obs_{self.obs_name[indobs]}.npz')
        np.savez(os.path.join(path, f'spectrum_obs_{self.obs_name[indobs]}.npz'), **self.obs_data[indobs])


    def _save_all_observations(self, path) -> None:
        '''
        Method to save all the observation files

        Parameters
        ----------
        path    (str | os.PathLike): Path to save the date to

        Authors: Allan Denis
        '''

        for indobs in range(self.n_obs):
            self._save_observation_file(path, indobs)


    def _load_adapted_observations_from_files(self, path: str | os.PathLike) -> None:
        '''
        Method to load adapted observations

        Parameters
        ----------
        path : str | os.PathLike
            DESCRIPTION.

        Authors: Simon Petrus, Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        '''

        obs_files = glob.glob(str(path) + '/spectrum_obs_*.npz')

        if len(obs_files) == 0:
            msg = f' No observation file in {path}.' + ' Make sure your observation files have the format spectrum_obs_{obs_name}.npz.'
            self._logger.error(msg)
            raise ForMoSAError(msg)

        if len(obs_files) != self.n_obs:
            msg = f' The number of files in the folder {path} does not correspond to the number of observations. Using only observations with the right name {obs_files}.'
            self._logger.warning(msg)

        missing_files = []
        for indobs in range(self.n_obs):
            obs_file = str(os.path.join(path, f'spectrum_obs_{self.obs_name[indobs]}.npz'))
            self._logger.debug(f'< Load observation file {obs_file}')

            if not(Path(obs_file).exists()):
                missing_files.append(obs_file)
            else:
                self._obs_data[indobs] = dict(np.load(os.path.join(path, obs_file), allow_pickle=True))
                # For some reasons, after loading the data with this method, the dictionary self._obs_data[indobs][component_type] is transformed into an array containing a dictionary,
                # so we use the following lines to transform it back into a dictionary
                if 'spectro' in self._obs_data[indobs] and isinstance(self._obs_data[indobs]['spectro'], np.ndarray) and self._obs_data[indobs]['spectro'].dtype == object:
                    self._obs_data[indobs]['spectro'] = self._obs_data[indobs]['spectro'].item()
                if 'photo' in self._obs_data[indobs] and isinstance(self._obs_data[indobs]['photo'], np.ndarray) and self._obs_data[indobs]['photo'].dtype == object:
                    self._obs_data[indobs]['photo'] = self._obs_data[indobs]['photo'].item()



        if len(missing_files) > 0:
            msg = f" Observation files cannot be found : {', '.join(missing_files)}."
            self._logger.error(msg)
            raise ForMoSAError(msg)