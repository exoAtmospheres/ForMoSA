from enum import Enum
import astropy.units as u
from pathlib import Path

class WavelengthUnit(Enum):
    '''
    Enumeration of wavelength units used in ForMoSA

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
    nanometer = u.um
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


class FilterPath(Enum):
    '''
    Enumeration of Filter path used in ForMoSA

    Authors: Allan Denis
    '''
    Path = Path(__file__).parent.parent / 'Filters'

    @property
    def path(self) -> Path:
        return self.value

