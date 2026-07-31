import numpy as np

import ForMoSA.utils.spec as us
from ForMoSA.core.errors import ForMoSAError
from ForMoSA.core.enums import VsiniFunction
from ForMoSA.transform.observed import ObservedModel
from ForMoSA.observation.observation_base import Observation

class ApplyPhysicsEffects:
    '''
    Apply physics effects to a model given parameters.

    Notes
    -----
    Authors: Allan Denis
    '''

    @staticmethod
    def _apply_rv(observed_model: ObservedModel, rv_value: float) -> ObservedModel:
        '''
        Apply radial velocity transformation.

        Parameters
        ----------
        observed_model : ObservedModel
            Instance of class ObservedModel
        rv_value : float
            RV value

        Returns
        -------
        observed_model : ObservedModel
            Instance of class ObservedModel Doppler shifted

        Notes
        -----
        Authors: Allan Denis
        '''

        observed_model.wave, observed_model.flux, observed_model.res = us.doppler_fct(observed_model.wave, observed_model.flux, observed_model.res, rv_value)

        return observed_model

    @staticmethod
    def _apply_vsini(observed_model: ObservedModel, vsini_value: float, ld_value: float, vsini_function: VsiniFunction) -> ObservedModel:
        '''
        Apply v.sini transformation.

        Parameters
        ----------
        observed_model : ObservedModel
            Instance of class ObservedModel
        vsini_value : float
            v.sini value
        ld_value : float
            limbd darkening value
        vsini_function : VsiniFunction
            Function used for the v.sini

        Returns
        -------
        observed_model : ObservedModel
            Instance of class ObservedModel rotationally broadened

        Notes
        -----
        Authors: Allan Denis
        '''

        observed_model.flux, observed_model.res = us.vsini_fct(observed_model.wave, observed_model.flux, observed_model.res, ld_value, vsini_value, vsini_function.value)

        return observed_model

    @staticmethod
    def _apply_reddening(observed_model: ObservedModel, av_value: float) -> np.ndarray:
        '''
        Apply redening transformation.

        Parameters
        ----------
        observed_model : ObservedModel
            Instance of class ObservedModel
        av_value : float
            Extinction value

        Returns
        -------
        observed_model : ObservedModel
            Instance of class ObservedModel redenned

        Notes
        -----
        Authors: Allan Denis
        '''

        observed_model.flux = us.reddening_fct(observed_model.wave, observed_model.flux, av_value)

        return observed_model

    @staticmethod
    def _apply_cpd(observed_model: ObservedModel, bb_T_value: float, bb_R_value: float, d_value: float) -> np.ndarray:
        '''
        Apply CPD effect.

        Parameters
        ----------
        observed_model : ObservedModel
            Instance of class ObservedModel
        bb_T_value : float
            Black body temperature value
        bb_R_value : float
            Black body radius value
        d_value : float
            Distance value

        Returns
        -------
        (ObservedModel): Instance of class ObservedModel with a CPD effect

        Notes
        -----
        Authors: Allan Denis
        '''

        observed_model.flux = us.bb_cpd_fct(observed_model.wave, observed_model.flux, d_value, bb_T_value, bb_R_value)

        return observed_model

    @staticmethod
    def _apply_scaling(observed_model: ObservedModel, r_value: float, d_value: float) -> np.ndarray:
        '''
        Apply scaling.

        Parameters
        ----------
        observed_model : ObservedModel
            Instance of class ObservedModel
        r_value : float
            Radius value
        d_value : float
            Distance value

        Returns
        -------
        flux_transformed : np.ndarray
            Scaled model

        Notes
        -----
        Authors: Allan Denis
        '''

        observed_model.flux, ck = us.calc_ck(observed_model.flux, [], [], r_value, d_value)
        observed_model.scaling = 'physical'

        return observed_model

class ApplyObservationEffects:

    @staticmethod
    def _apply_resolution_decreasing(observed_model: ObservedModel, obs: Observation) -> ObservedModel:
        '''
        Apply resolution decreasing.

        Parameters
        ----------
        observed_model : ObservedModel
            Instance of class ObservedModel to transform
        obs : Observation
            Observation

        Returns
        -------
        (ObservedModel): Instance of class ObservedModel degraded in resolution

        Notes
        -----
        Authors: Allan Denis
        '''

        if obs.res.all() == 0:
            return observed_model

        if len(observed_model.wave) != len(obs.wave):
            flux = us.resolution_decreasing(observed_model.wave, observed_model.flux, observed_model.res, obs.wave, obs.res)

            return ObservedModel(obs.wave, flux, obs.res)
        return observed_model

    @staticmethod
    def _apply_scaling(observed_model: ObservedModel, obs: Observation, bounds: tuple[float, float] = (-float('inf'), float('inf'))) -> ObservedModel:
        '''
        Apply analytical scaling between a model and observations.

        Parameters
        ----------
        observed_model : ObservedModel
            Instance of class ObservedModel to transform
        obs : Observation
            Observation
        bounds : tuple[float, float]
            Bounds for the Least Squares

        Returns
        -------
        (ObservedModel): Instance of class ObservedModel scaled to the data

        Notes
        -----
        Authors: Allan Denis
        '''

        if observed_model.scaling == 'analytic':
            observed_model.flux, ck = us.calc_ck(observed_model.flux, obs.flux, obs.err, 0, 0, analytic='yes', bounds = bounds)

        return observed_model

    @staticmethod
    def _hc_modeling(observed_model: ObservedModel, obs: Observation, bounds: tuple[float, float] = (-float('inf'), float('inf'))) -> ObservedModel:
        '''
        High-contrast modeling: solve flux_obs = sum_i coeffs_i * components_i
        where components_0 is model and components_i (i >= 1) is stellar speckles or systematics.

        Parameters
        ----------
        observed_model : ObservedModel
            Instance of class ObservedModel to transform
        obs : Observation
            Observation
        bounds : tuple[float, float]
            Bounds for the least squares

        Returns
        -------
        (ObservedModel): Instance of class ObservedModel high-contrast modelled

        Notes
        -----
        Authors: Allan Denis
        '''

        # =================================
        # Initial checks
        # =================================
        if not obs.hc_mode:
            raise ForMoSAError('Your observation needs to be in high-contrast mode in order to be able to use high-contrast modeling' )

        # =================================
        # Consistency between model and obs
        # =================================
        wave_cont, res_cont = obs.wave_cont, obs.res_cont
        wave, res = obs.wave, obs.res
        if len(wave) != len(observed_model.wave):
            raise ForMoSAError(f'Wrong length for model wavelength: {len(observed_model.wave)}. Expected the same length as the observations: {len(wave)}' )

        # High-contrast modeling requires wave_cont and res_cont for continuum estimation.
        # These must be set via the config (wav_cont / res_cont in [config_adapt]).
        if wave_cont is None or wave_cont == 'NA':
            raise ForMoSAError(
                'wave_cont is not set on the observation. '
                'High-contrast modeling requires wav_cont to be defined in the config file (e.g. wav_cont = 1.4, 1.8).'
            )

        if res_cont is None or res_cont == 'NA':
            raise ForMoSAError(
                'res_cont is not set on the observation. '
                'High-contrast modeling requires res_cont to be defined in the config file (e.g. res_cont = 100).'
            )

        observed_model.flux_cont = us.continuum_estimate(wave, observed_model.flux * obs.transm, res, wave_cont, res_cont)

        # =================================
        # Build components for linear fit
        # =================================
        components, fixed_coeffs, labels = us.build_linear_components(
            observed_model.flux,
            transm=obs.transm,
            flx_cont_obs=obs.flux_cont,
            flx_cont_mod=observed_model.flux_cont,
            star_flx_obs=obs.star_flux,
            star_flx_cont_obs=obs.star_flux_cont,
            system_obs=obs.system,
            analytic='yes',
        )

        # =================================
        # Solve linear model
        # =================================
        result = us.fit_linear_model(
            components=components,
            flx_obs=obs.flux,
            err_obs=obs.err,
            bounds=bounds,
            fixed_coeffs=fixed_coeffs
        )

        observed_model.flux, observed_model.components = result['reconstructed'][0], result['reconstructed'][1: len(components)]

        return observed_model


