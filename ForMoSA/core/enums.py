from enum import Enum
import astropy.units as u

class WavelengthUnit(Enum):
    '''
    Enumeration of wavelength units used in ForMoSA with accepted aliases

    Authors: Allan Denis
    '''

    ANGSTROM = u.AA
    NANOMETER = u.nm
    MICROMETER = u.um

    # Aliases for ANGSTROM
    angstrom = u.AA
    Angstrom = u.AA
    AA = u.AA

    # Aliases for NANOMETER
    Nanometer = u.nm
    nanometer = u.nm
    nm = u.nm

    # Aliases for MICROMETER
    Micrometer = u.um
    micrometer = u.um
    um = u.um

    @property
    def unit(self) -> u.Unit:
        return self.value

class FluxUnit(Enum):
    '''
    Enumeration of flux units used in ForMoSA

    Authors: Allan Denis
    '''

    FLAM = u.erg / (u.s * u.cm**2 * u.AA)
    FNU = u.erg / (u.s * u.cm**2 * u.Hz)
    JY = u.Jy

    @property
    def unit(self) -> u.Unit:
        return self.value

class DataUnit(Enum):
    '''
    Enumeration of data units used in ForMoSA

    Authors: Allan Denis
    '''

    COUNTS = 'counts'
    ELECTRONS = 'electrons'
    ADU = 'adu'

    @property
    def unit(self) -> str:
        return self.value

class ObservationType(Enum):
    '''
    Enumeration of observation types used in ForMoSA with accepted aliases

    Authors: Allan Denis
    '''

    SPECTROSCOPIC = 'Spectroscopic'
    PHOTOMETRIC = 'Photometric'

    # Aliases for spectro
    spectroscopic = 'Spectroscopic'
    SPECTRO = 'Spectroscopic'
    spectro = 'Spectroscopic'

    # Aliases for photo
    photometric = 'Photometric'
    PHOTO = 'Photometric'
    photo = 'Photometric'

    @property
    def obstype(self) -> str:
        return self.value

class ObservationKeys(Enum):
    '''
    Enumeration of observation keys used in ForMoSA with accepted aliases

    Examples
    --------
    >>> WAVELENGTH = ObservationKeys.WAVELENGTH
    >>> WAVELENGTH.canonical = "WAVELENGTH"
    >>> WAVELENGTH.aliases = "WAVE", "WAV", "LAMBDA"

    So the user can use "WAVELENGTH", "WAVE", "WAV" or "LAMBDA" to call the wavelength array of the observations

    Authors: Allan Denis
    '''

    WAVELENGTH = ("WAVELENGTH", ["WAVELENGTH", "WAVE", "WAV", "LAMBDA"])
    FLUX = ("FLUX", ["FLUX", "FLX"])
    ERROR = ("ERROR", ["ERROR", "ERR", "SIGMA"])
    FACILITY = ("FACILITY", ["FACILITY", "FAC", 'Facility'])
    INSTRUMENT = ("INSTRUMENT", ["INSTRUMENT", "INS"])
    FILTER_ID = ("FILTER_ID", ["FILTER_ID", "FILTER", "FILT", "FILT_ID"])
    RESOLUTION = ("RESOLUTION", ["RESOLUTION", "RES"])
    COVARIANCE = ("COVARIANCE", ["COVARIANCE", "COV"])
    TRANSMISSION = ("TRANSMISSION", ["TRANSMISSION", "TRANSM"])
    STAR_FLUX = ("STAR_FLUX", ["STAR_FLUX", "STAR_FLX"])
    SYSTEMATICS = ("SYSTEMATICS", ["SYSTEMATICS", "SYS"])
    WAVELENGTH_UNIT = ("WAVELENGTH_UNIT", ["WAVELENGTH_UNIT", "UNIT", "WAVE_UNIT"])
    FLUX_CONT = ("FLUX_CONT", ["FLUX_CONT"])
    STAR_FLUX_CONT = ("STAR_FLUX_CONT", ["STAR_FLUX_CONT"])
    WAVE_CONT = ("WAVE_CONT", ["WAVE_CONT"])
    RES_CONT = ("RES_CONT", ["RES_CONT"])

    def __init__(self, canonical: str, aliases: list[str]):
        self.canonical = canonical
        self.aliases = [a.upper() for a in aliases]

    # -------------------
    # Required keys
    # -------------------

    # Required parameters for all types of observations
    @classmethod
    def required_common(cls):
        return [cls.WAVELENGTH, cls.FLUX]

    # Required parameters for spectroscopic observations
    @classmethod
    def required_spectroscopic(cls):
        return [cls.RESOLUTION]

    # Required parameters for photometric observations
    @classmethod
    def required_photometric(cls):
        return [cls.ERROR, cls.FILTER_ID, cls.FACILITY, cls.INSTRUMENT]

    # ------------------
    # Validations
    # ------------------

    @classmethod
    def validate_photometric(cls, present: set[str]) -> list[str]:
        '''
        Method to check that observation keys contains required photometric keys

        Parameters
        ----------
        present : set[str]
            set of canonical observation keys

        Returns
        -------
        list[str]
            List of required photometric keys absent

        Authors: Allan Denis
        '''

        missing = [key.canonical for key in cls.required_common() if key.canonical not in present]
        missing.extend([key.canonical for key in cls.required_photometric() if key.canonical not in present])

        return missing

    @classmethod
    def validate_spectroscopic(cls, present: set[str]):
        '''
        Method to check that observation keys contains required spectroscopic keys

        Parameters
        ----------
        present : set[str]
            set of canonical observation keys

        Returns
        -------
        list[str]
            List of required spectroscopic keys absent

        Authors: Allan Denis
        '''

        missing = [key.canonical for key in cls.required_common() if key.canonical not in present]
        missing.extend([key.canonical for key in cls.required_spectroscopic() if key.canonical not in present])

        # Error OR covariance required
        has_error = cls.ERROR.canonical in present
        has_cov = cls.COVARIANCE.canonical in present

        if not (has_error or has_cov):
            missing.append(f"{cls.ERROR.canonical} or {cls.COVARIANCE.canonical}")

        return missing

    # ------------------
    # Utility
    # ------------------

    @classmethod
    def from_string(cls, value: str):
        value = value.upper()
        for key in cls:
            if value in key.aliases:
                return key
        raise ValueError(f"Unknown observation key: {value}")

class PriorType(Enum):
    '''
    Enumeration of prior types used in ForMoSA

    Authors: Allan Denis
    '''

    UNIFORM = 'uniform'
    GAUSSIAN = 'gaussian'
    LOG_UNIFORM = 'log_uniform'
    CONSTANT = 'constant'

    @property
    def is_gaussian(self) -> bool:
        return self == PriorType.GAUSSIAN

    @property
    def is_uniform(self) -> bool:
        return self == PriorType.UNIFORM

    @property
    def is_log_uniform(self) -> bool:
        return self == PriorType.LOG_UNIFORM

    @property
    def is_constant(self) -> bool:
        return self == PriorType.CONSTANT

    @property
    def priortype(self) -> str:
        return self.value

class VsiniFunction(Enum):
    '''
    Enumeration of vsini calculation functions used in ForMoSA

    Authors: Allan Denis
    '''

    RotBroad = 'RotBroad'
    FastRotBroad = 'FastRotBroad'
    Accurate = 'Accurate'
    AccurateFast = 'AccurateFast'

    @property
    def function(self) -> str:
        return self.value

class ParameterKind(Enum):
    '''
    Enumeration of parameters names used in ForMoSA

    Authors: Allan Denis
    '''

    RADIUS = 'r'
    DISTANCE = 'd'
    AV = 'av'
    BB_T = 'bb_T'
    BB_R = 'bb_R'
    ALPHA = 'alpha'
    RV = 'rv'
    VSINI = 'vsini'
    LD = 'ld'
    GRID = 'grid'

    # Radius aliases
    radius = 'r'
    R = 'r'
    r = 'r'

    # Distance aliases
    distance = 'd'
    D = 'd'
    d = 'd'

    # Extinction alises
    av = 'av'
    EXTINCTION = 'av'
    extinction = 'av'

    # Black body alises
    bb_T = 'bb_T'
    bb_R = 'bb_R'

    # Alpha aliases
    alpha = 'alpha'

    # Radial velocity aliases
    rv = 'rv'

    # Vsini aliases
    vsini = 'vsini'

    # Limb Darkening aliases
    ld = 'ld'

    # Grid aliases
    grid = 'grid'

    @property
    def kind(self) -> str:
        return self.value

class LogLikelihoodType(Enum):
    '''
    Enumeration of log-likelihood types used in ForMoSA

    Authors: Allan Denis
    '''

    CHI2 = 'chi2'
    CHI2_COVARIANCE = 'chi2_covariance'
    CHI2_NOISESCALING = 'chi2_noisescaling'
    CHI2_NOISESCALING_COVARIANCE = 'chi2_noisescaling_covariance'
    CCF_ZUCKER = 'CCF_Zucker'
    CCF_BROGI = 'CCF_Brogi'
    CCF_CUSTOM = 'CCF_custom'

    @property
    def loglike(self) -> str:
        return self.value

class NestedAlgorithm(Enum):
    '''
    Enumeration of nested sampling algorithms used in ForMoSA

    Authors: Allan Denis
    '''

    NESTLE = 'nestle'
    ULTRANEST = 'ultranest'
    PYMULTINEST = 'pymultinest'

    @property
    def algo(self) -> str:
        return self.value
