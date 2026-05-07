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

    Notes
    -----
    Authors: Allan Denis
    '''

    @staticmethod
    def _attributes_to_dict(**kwargs) -> dict:
        '''
        Mechanical conversion of attributes to a dictionary.

        kwargs: Keyword arguments

        Returns
        -------
        dict
            Dictionnary representation of the attributes

        Notes
        -----
        Authors: Allan Denis
        '''

        return {k: np.asarray(v) for k, v in kwargs.items() if v is not None}

    @staticmethod
    def _normalize_keys(keys: Iterable[str]) -> dict[str, str]:
        '''
        Normalize input keys to canonical ObservationKey names.

        Parameters
        ----------
        keys : iterable
            Column names (FITS table) or dictionary keys.

        Returns
        -------
        Mapping {canonical_key: actual_key_in_input}

        Examples
        --------
        >>> normalize_keys(["wave", "flux", "err", "instrument"]) --> {'WAVELENGTH': 'wave', 'FLUX': 'flux', 'ERROR': 'err', 'INSTRUMENT': 'instrument'}

        Notes
        -----
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
        path : str | os.PathLike
            Path of the data (Fits file)
        logger : logging.Logger
            Logger
        log_level : str
            Level of the Logger
        **kwargs                          : Additional arguments

        Returns
        -------
        Observation
            Instance of class Observation

        Notes
        -----
        Authors: Allan Denis
        '''

        logger = logger if logger is not None else setup_logging(log_level, name='Observation loader')

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
        data : Mapping[str, np.ndarray]
            Data
        logger : logging.Logger
            Logger
        log_level : str
            Level of the Logger
        **kwargs                        : Additional arguments

        Returns
        -------
        Observation
            Instance of class Observation

        Notes
        -----
        Authors: Allan Denis
        '''

        logger = logger if logger is not None else setup_logging(log_level, name='Observation loader')
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

        Notes
        -----
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
        data : Mapping[str, np.ndarray]
            Data
        logger : logging.Logger
            Logger
        log_level : str
            Level of the Logger
        **kwargs                        : Additional arguments

        Returns
        -------
        Observation
            instance of class Observation

        Notes
        -----
        Authors: Allan Denis
        '''

        logger = logger if logger is not None else setup_logging(log_level, name='Observation loader')

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
        # Detect photometric observation
        # --------------------------------
        missing_photo = ObservationKeys.validate_photometric(set(normalized.keys()))
        is_photo = len(missing_photo) == 0

        # --------------------------------
        # Detect spectroscopic observation
        # --------------------------------
        missing_spectro = ObservationKeys.validate_spectroscopic(set(normalized.keys()))
        is_spectro = len(missing_spectro) == 0

        # --------------------------------------------------------------------
        # NEW: If FILTER_ID exists but is only a placeholder (e.g., "NA"),
        # treat the observation as NOT photometric (prevents SVO filter lookup).
        # --------------------------------------------------------------------
        if is_photo and ObservationKeys.FILTER_ID.canonical in normalized:
            raw_filter = np.asarray(data[normalized["FILTER_ID"]], dtype=str)

            # Normalize (strip spaces, uppercase)
            filt_norm = np.char.upper(np.char.strip(raw_filter.astype(str)))

            # Tokens meaning "no filter" / placeholder
            null_tokens = {"", "NA"}

            # Keep only "real" filter ids
            valid_mask = ~np.isin(filt_norm, list(null_tokens))
            has_any_valid_filter = bool(np.any(valid_mask))

            if not has_any_valid_filter:
                logger.warning(
                    f"Observations FITS has a FILT column with values {sorted(null_tokens)};"
                    " If Observations are spectroscopic, please remove the FILT column for future use."
                    " Treating this observation as spectroscopic."
                )
                is_photo = False
                # Recompute missing_photo for correct error reporting if we end up in the final else:
                missing_photo = ObservationKeys.validate_photometric(set(normalized.keys()))

        # =============================
        # Photometric observation
        # =============================
        if is_photo:
            # Facility, ins and filter_id
            facility = np.asarray(data[normalized["FACILITY"]], dtype=str)
            ins = np.asarray(data[normalized["INSTRUMENT"]], dtype=str)
            filter_id = np.asarray(data[normalized["FILTER_ID"]], dtype=str)

            logger.info(f'    Detected photometric observation with filter {np.unique(filter_id)}')

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
                logger=logger,
            )

        # =============================
        # Spectroscopic observation
        # =============================
        elif is_spectro:
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
                wave_cont = (str(data[normalized['WAVE_CONT']]) if ObservationKeys.WAVE_CONT.canonical in normalized else kwargs.get('wave_cont', 'NA'))
            except TypeError:  # wave_cont is None
                wave_cont = 'NA'

            try:
                res_cont = (float(data[normalized['RES_CONT']]) if ObservationKeys.RES_CONT.canonical in normalized else kwargs.get('res_cont', 'NA'))
            except TypeError:  # res_cont is None
                res_cont = 'NA'

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
                logger=logger,
            )
            obs._flux_cont = flux_cont
            obs._star_flux_cont = star_flux_cont
            obs._res_cont = res_cont
            obs._wave_cont = wave_cont

        else:
            raise ForMoSAError(
                "Invalid observation: keys do not satisfy spectroscopic nor photometric requirements\n"
                f"Missing spectroscopic keys: {missing_spectro}\n"
                f"Missing photometric keys: {missing_photo}",
                logger
            )

        return obs

    @staticmethod
    def _extract_vector_series(data: Mapping[str, np.ndarray], key: ObservationKeys, **kwargs) -> None | np.ndarray:
        '''
        Extract columns like PREFIX1, PREFIX2, ... and stack them.

        Examples
        --------
        >>> systematics_array = ObservationFactory._extract_vector_series(data, ObservationKeys.SYSTEMATICS)
        >>> np.ndarray([systematics1, systematics2, systematics3, ...])

        Parameters
        ----------
        data : Mapping[str, np.ndarray]
            Data
        key : ObservationKeys
            Key of the column we want to extract (ObservationKeys.STAR_FLUX, ObservationKeys.SYSTEMATICS)
        **kwargs                         : Additional arguments

        Returns
        -------
        None | np.ndarray
            Stacked columns

        Notes
        -----
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
