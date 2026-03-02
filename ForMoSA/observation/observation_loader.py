import os
import logging
from astropy.io import fits
import numpy as np
from collections.abc import Iterable, Mapping

from ForMoSA.core.errors import ForMoSAError
from ForMoSA.core.loggings import setup_logging
from ForMoSA.core.enums import ObservationKeys, WavelengthUnit
from ForMoSA.observation.observation_base import Observation
from ForMoSA.observation.observation_spectroscopy import SpectralObservation
from ForMoSA.observation.observation_photometry import PhotometryObservation
from ForMoSA.utils import misc

class ObservationLoader:
    '''
    Class responsible for observation loading from various inputs format

    Authors: Allan Denis
    '''

    @staticmethod
    def _attributes_to_dict(**kwargs) -> dict:
        '''
        Mechanical conversion of attributes to a dictionary.

        kwargs: Keyword arguments

        Returns
        -------
        dict: Dictionnary representation of the attributes

        Authors: Allan Denis
        '''

        return {k: np.asarray(v) for k, v in kwargs.items() if v is not None}

    @staticmethod
    def _normalize_keys(keys: Iterable[str]) -> dict[str, str]:
        '''
        Normalize input keys to canonical ObservationKey names.

        Parameters
        ----------
        keys          (iterable): Column names (FITS table) or dictionary keys.

        Returns
        -------
        Mapping {canonical_key: actual_key_in_input}

        Usage: normalize_keys(["wave", "flux", "err", "instrument"]) --> {'WAVELENGTH': 'wave', 'FLUX': 'flux', 'ERROR': 'err', 'INSTRUMENT': 'instrument'}

        Authors: Allan Denis
        '''

        if not isinstance(keys, Iterable):
            raise ForMoSAError("keys must be an iterable of strings")

        normalized: dict[str, str] = {}

        # Uppercase mapping
        upper_keys = {k.upper(): k for k in keys}

        for obs_key in ObservationKeys:
            for alias in obs_key.aliases:
                if alias in upper_keys:
                    normalized[obs_key.canonical] = upper_keys[alias]
                    break

        return normalized

    @staticmethod
    def _from_fits(path: str | os.PathLike, logger: logging.Logger | None = None, log_level: str='INFO', **kwargs) -> Observation:
        '''
        Create an Observation from a FITS file.

        Parameters
        ----------
        path           (str | os.PathLike): Path of the data (Fits file)
        logger            (logging.Logger): Logger
        log_level                    (str): Level of the Logger
        **kwargs                          : Additional arguments

        Returns
        -------
        Observation: Instance of class Observation

        Authors: Allan Denis
        '''

        logger = logger or setup_logging(log_level, name='Observation loader')

        if not str(path).lower().endswith(".fits"):
            raise ForMoSAError(f'{path} is not a FITS file')

        with fits.open(path) as hdul:
            if len(hdul) < 2:
                raise ForMoSAError(f'{path} does not contain a data extension')

            logger.info(f'    Loading Observation from FITS file: {path}')
            data = misc.from_recarray_to_dic(hdul[1].data)

            return ObservationLoader._from_mapping(data=data, logger= logger, **kwargs)

    @staticmethod
    def _from_data(data: Mapping[str, np.ndarray], logger: logging.Logger | None = None, log_level: str='INFO', **kwargs) -> Observation:
        '''
        Create an Observation from an in-memory dictionary.

        Parameters
        ----------
        data  (Mapping[str, np.ndarray]): Data
        logger          (logging.Logger): Logger
        log_level                  (str): Level of the Logger
        **kwargs                        : Additional arguments

        Returns
        -------
        Observation: Instance of class Observation

        Authors: Allan Denis
        '''

        logger = logger or setup_logging(log_level, name='Observation loader')
        logger.info('    Creating Observation from data')
        return ObservationLoader._from_mapping(data=data, logger=logger, **kwargs)

    @staticmethod
    def _from_attributes(**kwargs) -> Observation:
        '''
        Create an Observation from a list of attributes. If non keyword arguments are passed, raise an error.

        Parameters
        ----------
        **kwargs : Keyword attributes

        Returns
        -------
        Observation

        Authors: Allan Denis
        '''

        # Retrieve logger arguments
        logger, log_level = kwargs.pop('logger', None), kwargs.pop('log_level', 'INFO')
        logger = logger or setup_logging(log_level, name='Observation loader')
        logger.info('    Loading observation from attributes')

        # Dictionnary representation of the data
        data = ObservationLoader._attributes_to_dict(**kwargs)

        return ObservationLoader._from_mapping(data=data, logger=logger, **kwargs)

    @staticmethod
    def _from_mapping(data: Mapping[str, np.ndarray], logger: logging.Logger | None = None, log_level: str='INFO', **kwargs) -> Observation:
        '''
        Core logic shared by FITS and in-memory creation.

        Parameters
        ----------
        data  (Mapping[str, np.ndarray]): Data
        logger          (logging.Logger): Logger
        log_level                  (str): Level of the Logger
        **kwargs                        : Additional arguments

        Returns
        -------
        Observation: instance of class Observation

        Authors: Allan Denis
        '''

        logger = logger or setup_logging(log_level, name='Observation loader')

        if not isinstance(data, Mapping):
            raise ForMoSAError('data must be a mapping (dict-like)', logger)

        # --------------------------------
        # Retrieve and normalize keys
        # --------------------------------
        keys = data.keys()
        normalized = ObservationLoader._normalize_keys(keys)

        # --------------------------------
        # Check required common keys
        # --------------------------------
        missing = [key.canonical for key in ObservationKeys.required_common() if key.canonical not in normalized]
        if missing:
            raise ForMoSAError(f" Missing required observation keys: {', '.join(missing)}", logger)

        # --------------------------------
        # Extract common data
        # --------------------------------
        wave = data[normalized["WAVELENGTH"]]
        flux = data[normalized["FLUX"]]

        # --------------------------------
        # Specific case for units
        # --------------------------------
        if ObservationKeys.WAVELENGTH_UNIT.canonical not in normalized:
            native_unit = WavelengthUnit.MICROMETER
            logger.warning(f'Wavelength unit not specified for observation. Assuming {native_unit.unit}')
        else:
            unit_value = np.unique(np.asarray(data[normalized["WAVELENGTH_UNIT"]], dtype=str))[0]
            native_unit = WavelengthUnit[str(unit_value)]

        # --------------------------------
        # Detect spectroscopic observation
        # --------------------------------
        is_spectro = (ObservationKeys.RESOLUTION.canonical in normalized and np.any(np.array(data[normalized["RESOLUTION"]]) > 0))

        # =============================
        # Spectroscopic observation
        # =============================
        if is_spectro:

            missing_spectro = ObservationKeys.validate_spectroscopic(set(normalized.keys()))
            if missing_spectro:
                raise ForMoSAError(f"Missing required observation keys: {', '.join(missing_spectro)}", logger)

            # Facility and instrument
            for canonical, key in zip([ObservationKeys.FACILITY.canonical, ObservationKeys.INSTRUMENT.canonical], ['FACILITY', 'INSTRUMENT']):
                if canonical not in set(normalized.keys()):
                    logger.warning(f"Key {key} not in observation keys. Setting it to 'unknown'")
                    if key == 'FACILITY':
                        facility = np.array(['unknown'] * len(wave))
                    elif key == 'INSTRUMENT':
                        ins = np.array(['unknown'] * len(wave))
                else:
                    if key == 'FACILITY':
                        facility = np.asarray(data[normalized["FACILITY"]], dtype=str)
                    elif key == 'INSTRUMENT':
                        ins = np.asarray(data[normalized["INSTRUMENT"]], dtype=str)

            logger.info(f'    Detected spectroscopic observation with instruments {np.unique(facility)}/{np.unique(ins)}')

            # Resolution
            res = data[normalized["RESOLUTION"]]

            # ----------------------------
            # Optional inputs
            # ----------------------------

            # Error / Covariance
            err = (data[normalized["ERROR"]] if ObservationKeys.ERROR.canonical in normalized else None)
            cov = (data[normalized["COVARIANCE"]] if ObservationKeys.COVARIANCE.canonical in normalized else None)

            if err is None:
                err = np.sqrt(np.diag(cov))

            # Transmission
            transm = (data[normalized["TRANSMISSION"]] if ObservationKeys.TRANSMISSION.canonical in normalized else None)

            # Stellar flux
            star_flux = ObservationLoader._extract_vector_series(data, ObservationKeys.STAR_FLUX, **kwargs)

            # Systematics
            system = ObservationLoader._extract_vector_series(data, ObservationKeys.SYSTEMATICS)

            # Continuums
            flux_cont = (data[normalized['FLUX_CONT']] if ObservationKeys.FLUX_CONT.canonical in normalized else None)
            star_flux_cont = (data[normalized['STAR_FLUX_CONT']] if ObservationKeys.STAR_FLUX_CONT.canonical in normalized else None)

            # Wavelength and resolution for the continuum
            try:
                wave_cont = (str(data[normalized['WAVE_CONT']]) if ObservationKeys.WAVE_CONT.canonical in normalized else kwargs.get('wave_cont', None))
            except TypeError:  # wave_cont is None
                wave_cont = None

            try:
                res_cont = (float(data[normalized['RES_CONT']]) if ObservationKeys.RES_CONT.canonical in normalized else kwargs.get('res_cont', None))
            except TypeError:  # res_cont is None
                res_cont = None

            obs = SpectralObservation(
                wave=wave,
                flux=flux,
                err=err,
                res=res,
                native_unit=native_unit,
                facility=facility,
                instrument=ins,
                cov=cov,
                transm=transm,
                star_flux=star_flux,
                system=system,
            )
            obs._flux_cont = flux_cont
            obs._star_flux_cont = star_flux_cont
            obs._res_cont = res_cont
            obs._wave_cont = wave_cont

        # =============================
        # Photometric observation
        # =============================
        else:
            missing_photo = ObservationKeys.validate_photometric(set(normalized.keys()))
            if missing_photo:
                raise ForMoSAError(f"Missing required observation keys: {', '.join(missing_photo)}", logger)

            # Facility, ins and filter_id
            facility = np.asarray(data[normalized["FACILITY"]], dtype=str)
            ins = np.asarray(data[normalized["INSTRUMENT"]], dtype=str)
            filter_id = np.asarray(data[normalized["FILTER_ID"]], dtype=str)

            # if len(facility) != 1:
            #     raise ForMoSAError(f'Wrong length for facility: {len(facility)}. You must provide only one facility')
            # if len(ins) !=1:
            #     raise ForMoSAError(f'Wrong length for instrument {len(ins)}. You must provide only one instrument')
            # if len(filter_id) != 1:
            #     raise ForMoSAError(f'Wrong length for filter_id: {len(filter_id)}. You must provide only one filter_id')

            #facility, ins, filter_id = facility[0], ins[0], filter_id[0]

            logger.info(f'    Detected photometric observation with filter {np.unique(facility)}/{np.unique(ins)}.{np.unique(filter_id)}')

            # Error
            err = data[normalized["ERROR"]]

            obs = PhotometryObservation(
                wave=wave,
                flux=flux,
                err=err,
                native_unit=native_unit,
                facility=facility,
                instrument=ins,
                filter_id=filter_id,
            )

        return obs

    @staticmethod
    def _extract_vector_series(data: Mapping[str, np.ndarray], key: ObservationKeys, **kwargs) -> None | np.ndarray:
        '''
        Extract columns like PREFIX1, PREFIX2, ... and stack them.

        Example usage: systematics_array = ObservationFactory._extract_vector_series(data, ObservationKeys.SYSTEMATICS)
        -> np.ndarray([systematics1, systematics2, systematics3, ...])

        Parameters
        ----------
        data   (Mapping[str, np.ndarray]): Data
        key             (ObservationKeys): Key of the column we want to extract (ObservationKeys.STAR_FLUX, ObservationKeys.SYSTEMATICS)
        **kwargs                         : Additional arguments

        Returns
        -------
        None | np.ndarray: Stacked columns

        Authors: Allan Denis
        '''

        # Get the aliases corresponding to the key
        aliases = tuple(alias.upper() for alias in key.aliases)

        # Get the keys matching the input ObservationKey in chronological order
        # Careful with STAR_FLUX_CONT keywords
        matched_keys = sorted(k for k in data.keys() if k.upper().startswith(aliases) and not k.upper().endswith('_CONT'))

        if not matched_keys:
            return None

        # Retrieve 2D data
        vectors = [np.atleast_2d(data[k]).reshape(len(data[k]), -1) for k in matched_keys]
        return np.concatenate(vectors, axis=1)
