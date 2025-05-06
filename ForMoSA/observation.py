import numpy as np 
import os
from pathlib import Path
import glob
from astropy.io import fits
from ForMoSA.utils_spec import resolution_decreasing, continuum_estimate

class ForMoSAError(Exception):
    pass

class Observation(object):
    '''
    ForMoSA observation class, which provides easy access to an observation
    
    Parameters
    ----------
    observation_path (str | os.PathLike): Path to the observation file(s) (multiple observation files can exist)
    log_level (str): Log level of the handler, by default ``'info'`` for all important informations.
    '''
    
    def __init__(self, observation_path: str | os.PathLike, logger) -> None:
        self._observation_path = Path(str(observation_path).rstrip('*') + '*').expanduser()
        self._obs_data = dict()
        self._logger = logger
        
        # extract observation data
        for indobs in range(self.n_obs):
            self._extract_observation(indobs)
        
    ##################################################
    # Representation
    ##################################################
    
    def __repr__(self) -> str:
        return f'<Observation, n_obs = {self.n_obs}>'

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
        if len(files) == 0:  # No observation
            self._logger.error(f' No observation. {self.observation_path} does not contain any observation.')
            return ForMoSAError()
        else:
            return {i: file for i, file in enumerate(files)}
    
    @property  
    def obs_name(self):
        _obs_name = dict()
        for indobs, obs in self.obs_files.items():
            _obs_name[indobs] = os.path.splitext(os.path.basename(obs))[0]
        return _obs_name
    
    @property 
    def obs_data(self):
        return self._obs_data
    
    @property 
    def n_obs(self):
        return len(self.obs_files)
        
    
    ##################################################
    # Methods
    ##################################################
        
    def _extract_observation(self, indobs=0):
        """
        Extract the information from the observation file, including the wavelengths (um - vacuum), flux (W.m-2.um.1), errors (W.m-2.um.1), covariance (W.m-2.um.1)**2, spectral resolution, 
        instrument/filter name, transmission (Atmo+inst) and star flux (W.m-2.um.1). The wavelength range is define by the parameter "wav_for_adapt".

        Args:
            global_params  (object): Class containing each parameter
            obs_name          (str): Name of the current observation looping
            indobs            (int): Index of the current observation looping

        Returns:
            - obs_dict       (dict): Dictionay containing all the observationnal entries (photometry, spectroscopy and/or optional)

        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        """
        # Extraction
        self._logger.info(f' Current observation {self.obs_files[indobs]}.')
        
        with fits.open(self.obs_files[indobs]) as hdul:
            self._logger.debug(f'> Read file {self.obs_files[indobs]}.')

            missing_keys = []
            
            # Check that parameters are well defined
            # General parameters 
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
                
                msg = f" Keys missing : {', '.join(missing_keys)}"
                self._logger.critical(msg)
                raise ForMoSAError(msg)
            
            # Errors and covariances
            try: 
                err = hdul[1].data['ERR']
                cov = np.asarray([]) # Create an empty covariance matrix if not already present in the data (to not slow the inversion)
                self._logger.info(f' Your observation {self.obs_name[indobs]} contains an error vector.')
            except KeyError:
                missing_keys.append('ERR')
                try:
                    cov = hdul[1].data['COV']
                    err = np.sqrt(np.diag(np.abs(cov)))
                    self._logger.info(f' Your observation {self.obs_name[indobs]} contains a covariance matrix.')
                except KeyError:
                    missing_keys.append('COV')
                    msg = f" Keys missing : {', '.join(missing_keys)}"
                    self._logger.critical(msg)
                    raise ForMoSAError(msg)
                    
            # High-contrast parameters
            try: # Check for transmission
                transm = hdul[1].data['TRANSM']
                self._logger.info(f' Your observation {self.obs_name[indobs]} contains a transmission vector.')
            except:
                transm = np.asarray([])
            try: # Check for star flux
                star_flx = hdul[1].data['STAR_FLX1'][:,np.newaxis]
                is_star = True
            except:
                star_flx = np.asarray([])   
                is_star = False
            if is_star:
                i = 2
                while True: # In case there is multiple star flux (usually shifted to account for the PSF)
                    try:
                        star_flx = np.concatenate((star_flx, hdul[1].data['STAR_FLX' + str(i)][:,np.newaxis]),axis=1)
                        i += 1
                    except:
                        self._logger.info(f' Your observation {self.obs_name[indobs]} contains {i-1} star vectors.')
                        break
            
            # Additional systematics parameters (sometimes used for high-contrast data)
            try:
                is_system = True
                system = hdul[1].data['SYSTEMATICS1'][:,np.newaxis]
            except:
                is_system = False
                system = np.asarray([])
            if is_system:
                i = 2
                while True: # In case there is multiple systematics
                    try:
                        system = np.concatenate((system, hdul[1].data['SYSTEMATICS' + str(i)][:,np.newaxis]),axis=1)
                        i += 1
                    except:
                        self._logger.info(f' Your observation {self.obs_name[indobs]} contains {i-1} systematics vectors.')
                        break

            # Only take the covariance if you use the chi2_covariance likelihood function (will need to be change when new likelihood functions using the
            # covariance matrix will come)
            #if global_params.logL_type[indobs % len(global_params.logL_type)] != 'chi2_covariance' and global_params.logL_type[indobs % len(global_params.logL_type)] != 'chi2_noisescaling_covariance':
            #    cov = np.asarray([])

            # Filter the NaN and inf values
            self._logger.debug('> Detect NaN data.')
            nan_mod_ind = (~np.isnan(flx)) & (~np.isnan(err)) & (np.isfinite(flx)) & (np.isfinite(err))
            if len(cov) != 0:
                nan_mod_ind = (nan_mod_ind) & np.all(~np.isnan(cov), axis=0) & np.all(~np.isnan(cov), axis=1) & np.all(np.isfinite(cov), axis=0) & np.all(np.isfinite(cov), axis=1)
            if len(transm) != 0:
                nan_mod_ind = (nan_mod_ind) & (~np.isnan(transm)) & (np.isfinite(transm))
            if len(star_flx) != 0:
                for i in range(len(star_flx[0])):
                    nan_mod_ind = (nan_mod_ind) & (~np.isnan(star_flx.T[i])) & (np.isfinite(star_flx.T[i]))
            if len(system) != 0:
                for i in range(len(system[0])):
                    nan_mod_ind = (nan_mod_ind) & (~np.isnan(system.T[i])) & (np.isfinite(system.T[i])) 
            self._logger.info(f'> {len(nan_mod_ind[nan_mod_ind == False])} NaN data out of {len(flx)}.')
                    
            wav = wav[nan_mod_ind]
            flx = flx[nan_mod_ind]
            res = res[nan_mod_ind]
            ins = ins[nan_mod_ind]
            err = err[nan_mod_ind]
            if len(cov) != 0:
                cov = np.transpose(np.transpose(cov[nan_mod_ind])[nan_mod_ind])
                inv_cov = np.linalg.inv(cov) # Save only the inverse covariance to speed up the inversion
            else:
                inv_cov = np.asarray([])
            if len(transm) != 0 and len(star_flx) != 0:
                transm = transm[nan_mod_ind]
            if len(star_flx) != 0:
                star_flx = np.delete(star_flx, np.where(~nan_mod_ind), axis=0)
            if len(system) != 0:
                system = np.delete(system, np.where(~nan_mod_ind), axis=0)

            # - - - - - - - - - 
            
            # Separate photometry from spectroscopy
            mask_photo = (res == 0.0)
            
            self._logger.info(f' Your observation {self.obs_name[indobs]} contains {len(wav[~mask_photo])} spectroscopic points and {len(wav[mask_photo])} photometric points.')
            # Check-ups and warnings for negative values in the diagonal of the covariance matrix
            if len(wav[~mask_photo]) != 0 and any(np.diag(inv_cov) < 0):
                self._logger.critical(" Negative value(s) is(are) present on the diagonal of the covariance matrix.") 
                raise ForMoSAError
           
            # Observation dictionary
            obs_dict = {'wav_photo': wav[mask_photo], # Photometry part
                        'flx_photo': flx[mask_photo],
                        'err_photo': err[mask_photo],
                        'ins_photo': ins[mask_photo],
                        'wav_spectro': wav[~mask_photo], # Spectroscopy part
                        'flx_spectro': flx[~mask_photo],
                        'err_spectro': err[~mask_photo],
                        'res_spectro': res[~mask_photo],
                        'inv_cov': inv_cov, # Optional part
                        'transm': transm,
                        'star_flx': star_flx,
                        'system': system
                        }
            
            
            self._obs_data[indobs] = obs_dict
            
            
    def _adapt_observation(self, target_res_obs: np.ndarray, res_cont: np.ndarray, wav_cont: np.ndarray=[], star_continuum: str='NA', indobs: int=0):
        """
        Decrease the spectral resolution of the current observation and remove the continuum if necessary

        Args:
            target_res_obs     (str): Target resolution of the observation
            res_cont           (str): Resolution of the continuum
            wav_cont       (ndarray): Wavelength of the continuum
            hc_type            (str): High contrast function
            indobs             (int): Index of the current observation
            
        Returns:
            obs_data          (dict): Adapted current observation

        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        """

        # Decrease the resolution and remove the continuum if necessary
        obs_data = self.obs_data[indobs]
        obs_name = self.obs_name[indobs]

        # If we want to decrease the resolution of the spectroscopic data:
        self._logger.info(f' Target resolution for the data: {target_res_obs}.')
        self._logger.debug('> Decrease the resolution of the flux if necessary.')
        obs_data['flx_spectro'] = resolution_decreasing(obs_data['wav_spectro'],
                                                        obs_data['flx_spectro'],
                                                        obs_data['res_spectro'],
                                                        obs_data['wav_spectro'],
                                                        target_res_obs)
        
        if len(obs_data['transm']) > 0:
            self._logger.debug('> Decrease the resolution of the transmission if necessary..')
            obs_data['transm'] = resolution_decreasing(obs_data['wav_spectro'],
                                                       obs_data['transm'],
                                                       obs_data['res_spectro'],
                                                       obs_data['wav_spectro'],
                                                       target_res_obs)
        if len(obs_data['star_flx']) > 0:
            self._logger.debug('> Decrease the resolution of the star flux if necessary.')
            obs_data['star_flx'] = np.asarray([resolution_decreasing(obs_data['wav_spectro'],
                                                                     obs_data['star_flx'][:,i],
                                                                     obs_data['res_spectro'],
                                                                     obs_data['wav_spectro'],
                                                                     target_res_obs) for i in range(obs_data['star_flx'].shape[-1])]).T
        
        if len(obs_data['system']) > 0:
            self._logger.debug('> Decrease the resolution of the systematics if necessary.')
            obs_data['system'] = np.asarray([resolution_decreasing(obs_data['wav_spectro'],
                                                                   obs_data['system'][:,i],
                                                                   obs_data['res_spectro'],
                                                                   obs_data['wav_spectro'],
                                                                   target_res_obs) for i in range(obs_data['system'].shape[-1])]).T
    
    
        # Since the resolution of the observation might have change, we need to save the new one
        obs_data['res_spectro'] = target_res_obs
        
        # If we want to estimate and substract the continuum of the data:
        if res_cont != 'NA':
            self._logger.debug('> Substract the continuum to the data')
            
            obs_data['flx_spectro_cont'] = continuum_estimate(obs_data['wav_spectro'], 
                                                              obs_data['flx_spectro'], 
                                                              obs_data['res_spectro'], 
                                                              wav_cont, res_cont)
            
            # We want to remove the continuum of the star (generally if you do not use high contrast data)
            if star_continuum == 'remove':
                obs_data['flx_spectro'] -= obs_data['flx_spectro_cont']
                self._logger.info(f' {obs_name} will have a R = {res_cont} continuum removed using a {wav_cont} wavelength range')

            elif star_continuum == 'estimate': # We just want to estimate the continuum of the star (generally if you use high contrast data)
                self._logger.debug('> Estimate the continuum to the stellar data')
                obs_data['star_flx_cont'] = continuum_estimate(obs_data['wav_spectro'],
                                                            obs_data['star_flx'][:,len(obs_data['star_flx'][0]) // 2], # Continuum of the star on the central pixel
                                                            obs_data['res_spectro'],
                                                            wav_cont, res_cont)
    
        self._obs_data[indobs] = obs_data
        
    