from ForMoSA.core.errors import ForMoSAError
import ForMoSA.utils.logL_functions as logL_functions
from ForMoSA.observation.observation_base import Observation
from ForMoSA.transform.observed import ObservedModel, ObservedParameters
from ForMoSA.core.enums import ParameterKind, ObservationType, LogLikelihoodType
from ForMoSA.transform.apply_effects import ApplyPhysicsEffects, ApplyObservationEffects

class SpectralEffects:
    '''
    Physical effects applied to a spectroscopic model.

    Authors
    -------
    Allan Denis
    '''

    @staticmethod
    def _apply_physics(observed_model: ObservedModel, observed_params: ObservedParameters) -> ObservedModel:
        '''
        Apply the physics effects relevant to spectroscopy.

        Parameters
        ----------
        observed_model : ObservedModel
            Instance of class ObservedModel
        params : ObservedParameters
            Instance of class ObservedParameters

        Returns
        -------
        (ObservedModel): Instance of class ObservedModel transformed by the physics effects

        Authors
        -------
        Allan Denis
        '''

        # ======================
        # Initial checkins
        # ======================

        # Instance of observed_model
        if not isinstance(observed_model, ObservedModel):
            raise ForMoSAError(f' Wrong type for observed_model: {type(observed_model)}. Expected an ObservedModel')

        # Instance of observed_params
        if not isinstance(observed_params, ObservedParameters):
            raise ForMoSAError(f' Wrong type for observed_params: {type(observed_params)}. Expected an ObservedParameters')

        values_by_kind = observed_params.values_by_kind

        # ======================
        # RV
        # ======================

        if observed_params.has_kind(ParameterKind.RV):
            observed_model = ApplyPhysicsEffects._apply_rv(observed_model, values_by_kind[ParameterKind.RV])

        # ======================
        # v.sini
        # ======================

        if ParameterKind.VSINI in values_by_kind:
            vsini_param = observed_params.params_by_kind[ParameterKind.VSINI]

            observed_model = ApplyPhysicsEffects._apply_vsini(observed_model, values_by_kind[ParameterKind.VSINI], values_by_kind[ParameterKind.LD], vsini_param.vsini_function)

        # ======================
        # Reddening
        # ======================

        if ParameterKind.AV in values_by_kind:
            observed_model = ApplyPhysicsEffects._apply_reddening(observed_model, values_by_kind[ParameterKind.AV])

        # ======================
        # CPD
        # ======================

        if ParameterKind.BB_T in values_by_kind:
            if ParameterKind.BB_R not in values_by_kind:
                raise ForMoSAError(' Black Body radius is required when a Black Body temperature is given')

            if ParameterKind.DISTANCE not in values_by_kind:
                raise ForMoSAError(' Distance is required to add a CPD contribution')

            observed_model = ApplyPhysicsEffects._apply_cpd(observed_model, values_by_kind[ParameterKind.BB_T], values_by_kind[ParameterKind.BB_R], values_by_kind[ParameterKind.DISTANCE])

        # ======================
        # Scaling (R, D, alpha)
        # ======================

        if ParameterKind.DISTANCE in values_by_kind and ParameterKind.RADIUS in values_by_kind:
            observed_model = ApplyPhysicsEffects._apply_scaling(observed_model, values_by_kind.get(ParameterKind.RADIUS), values_by_kind.get(ParameterKind.DISTANCE))

        alpha = values_by_kind.get(ParameterKind.ALPHA, 1.0)
        observed_model.flux *= alpha

        return observed_model

    @staticmethod
    def _apply_observation(observed_model: ObservedModel, obs: Observation, bounds: tuple[float, float] = (-float('inf'), float('inf'))) -> ObservedModel:
        '''
        Apply the observational effects relevant to spectroscopy.

        Parameters
        ----------
        observed_model : ObservedModel
            Instance of class ObservedModel
        obs : Observation
            Instance of class Observation
        bounds : tuple[float, float]
            Bounds for the least squares

        Returns
        -------
        ObservedModel
            Instance of class ObservedModel transformed by the observational effects

        Authors
        -------
        Allan Denis
        '''

        # ======================
        # Initial checkins
        # ======================

        # Instance of observed_model
        if not isinstance(observed_model, ObservedModel):
            raise ForMoSAError(f' Wrong type for observed_model: {type(observed_model)}. Expected a ObservedModel')

        # Instance of observation
        if not isinstance(obs, Observation):
            raise ForMoSAError(f' Wrong obs type: {type(obs)}. Expected an Observation')

        # ObsType
        if obs.ObsType != ObservationType.SPECTROSCOPIC.value:
            raise ForMoSAError(f' Wrong observation type: {obs.ObsType}. Expected {ObservationType.SPECTROSCOPIC.value}')

        # ======================
        # Resolution decreasing
        # ======================

        if len(observed_model.wave) != len(obs.wave):
            observed_model = ApplyObservationEffects._apply_resolution_decreasing(observed_model, obs)

        # ======================
        # Scaling
        # ======================

        if observed_model.scaling == 'analytic' and not obs.hc_mode:
            observed_model = ApplyObservationEffects._apply_scaling(observed_model, obs)

        # ======================
        # High-contrast modeling
        # ======================

        if obs.hc_mode:
            observed_model = ApplyObservationEffects._hc_modeling(observed_model, obs, bounds=bounds)

        return observed_model

    @staticmethod
    def _compute_loglike(observed_model: ObservedModel, obs: Observation, logL_type: LogLikelihoodType) -> float:
        '''
        Compute the loglikelihood given an ObservedModel, an Observation and a loglikelihood function.

        Parameters
        ----------
        observed_model : ObservedModel
            Observed model
        obs : Observation
            Observation
        logL_type : LogLikelihoodType
            Loglikelihood function

        Returns
        -------
        float
            logL value

        Authors
        -------
        Simon Petrus, Matthieu Ravet and Allan Denis
        '''

        # ======================
        # Initial checks
        # ======================

        if not isinstance(observed_model, ObservedModel):
            raise ForMoSAError(f' Wrong type for model: {type(observed_model)}. Require an ObservedModel')

        if not isinstance(obs, Observation):
            raise ForMoSAError(f' Wrong type for observation: {type(obs)}. Require an Observation')

        if not isinstance(logL_type, LogLikelihoodType):
            raise ForMoSAError(f' Wrong type for logL_type {type(logL_type)}. Require a LogLikelihoodType')

        if obs.ObsType is not ObservationType.SPECTROSCOPIC.obstype:
            raise ForMoSAError(f' Wrong Observation type: {obs.ObsType}. Use a spectroscopic observation')

        if len(observed_model.flux) != len(obs.flux):
            raise ForMoSAError(f' Wrong length for model: {len(observed_model.flux)}. Expected the same length as the observations: {len(obs.flux)}')

        # Residuals and residuals with the componant only
        residuals = observed_model.residuals(obs.flux)
        residuals_componant = observed_model.residuals(obs.flux, component_only=True)

        # =====================
        # Compute loglikelihood
        # =====================

        logL_dict = {
            LogLikelihoodType.CHI2: lambda: logL_functions.logL_chi2(residuals, obs.err),
            LogLikelihoodType.CHI2_COVARIANCE: lambda: logL_functions.logL_chi2_covariance(residuals, obs.cov, obs.inv_cov),
            LogLikelihoodType.CCF_BROGI: lambda: logL_functions.logL_CCF_Brogi(residuals_componant, observed_model.flux),
            LogLikelihoodType.CCF_ZUCKER: lambda: logL_functions.logL_CCF_Zucker(residuals_componant, observed_model.flux),
            LogLikelihoodType.CCF_CUSTOM: lambda: logL_functions.logL_CCF_custom(residuals_componant, observed_model.flux, obs.err),
            LogLikelihoodType.CHI2_NOISESCALING: lambda: logL_functions.logL_chi2_noisescaling(residuals, obs.err),
            LogLikelihoodType.CHI2_NOISESCALING_COVARIANCE: lambda: logL_functions.logL_chi2_noisescaling_covariance(residuals, obs.cov, obs.inv_cov)
        }

        return logL_dict.get(logL_type, lambda: 0)()
