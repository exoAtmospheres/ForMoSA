from astroquery.svo_fps import SvoFps
import numpy as np
import matplotlib.pyplot as plt
import logging
import astropy.units as u
import astropy
from ForMoSA.ForMoSA_logging import setup_logging
from ForMoSA.ForMoSA_error import ForMoSAError
from ForMoSA.ForMoSA_enums import WavelengthUnit, FilterPath
from astropy.table import Table, MaskedColumn, Column
from pathlib import Path
from astropy.io import fits
import os

class Filter(object):
    '''
    Filter class providing access to the properties of the filter
    The faclity, instrument and filter_id has to be under the right format (see https://svo2.cab.inta-csic.es/theory/fps/ for more details)

    Parameters
    ----------
    facility   (str): Name of the facility ('Paranal', 'Keck', 'JWST', ...)
    instrument (str): Name of the instrument ('SPHERE', 'NIRC2', 'NIRCam', ...)
    filter_id  (str): ID of the filter ('IRDIS_B_H', 'Lp', F410M)

    Authors: Allan Denis and Mickael Bonnefoy
    '''

    def __init__(self, facility: str, instrument: str, filter_id: str, log_level: str = 'info', logger: logging.Logger = None):
        if logger == None:
            self._logger = setup_logging(level=log_level, name=__name__)
        else:
            self._logger = logger

        self._facility = facility
        self._instrument = instrument
        self._filter_id = filter_id

        self._data = []
        self._medata = []

        self._native_unit = WavelengthUnit.ANGSTROM
        self._display_unit = WavelengthUnit.ANGSTROM
        self._svo_filter_trans()

    ##################################################
    # Representation
    ##################################################

    def __repr__(self) -> str:
        return f'<Filter, name={self.name}, central wavelength = {self.central_wavelength}, range = [{self.wavelength_min.value} - {self.wavelength_max.value}] {self.wavelength.unit}>'


    def __format__(self) -> str:
        return self.__repr__()

    #################################################
    # Properties
    #################################################

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # General properties
    # # # # # # # # # # # # # # # # # # # # # # # # #

    @property
    def logger(self) -> logging.Logger:                   # Logger
        return self._logger

    @property
    def facility(self) -> str:                            # Facility for the filter (e.g. 'Paranal', 'Keck', 'JWST', ...)
        return self._facility

    @property
    def instrument(self) -> str:                          # Instrument used of the filter (e.g. 'SPHERE', 'NIRC2', 'NIRCam', ...)
        return self._instrument

    @property
    def filter_id(self) -> str:                           # ID of the filter (e.g. 'IRDIS_B_H', 'Lp', 'F410M', ...)
        return self._filter_id

    @property
    def name(self) -> str:                                # Full name of the filter (e.g. 'Paranal/SPHERE.IRDIS_B_H', 'Keck/NIRC2.Lp', 'JWST/NIRCam.F410M')
        return self.facility + '/' + self.instrument + '.' + self.filter_id

    @property
    def data(self) -> astropy.table.table.Table:          # Table containing the data of the filter (wavelength, transmission)
        return self._data

    @property
    def unit(self) -> u.core.PrefixUnit:                  # Unit to use for the wavelength
        return self._display_unit.unit

    @property
    def native_unit(self) -> u.core.PrefixUnit:           # Native unit of the wavelength (Angstrom)
        return self._native_unit.unit

    @property
    def metadata(self) -> astropy.table.table.Table:      # Table containing the metadata of the filter
        return self._metadata

    @property
    def wavelength(self) -> u.quantity.Quantity:          # Wavelength grid of the filter
        return (self.data['Wavelength'].data.filled(np.nan) * self.native_unit).to(self.unit)

    @property
    def transmission(self) -> np.ndarray:                 # Transmission of the filter
        return self.data['Transmission'].data.filled(np.nan)

    @property
    def fwhm(self) -> u.quantity.Quantity:                # FWHM
        return self.metadata['FWHM'][0] * self.unit

    @property
    def zero_point(self) -> np.float64:                   # Zero point
        return self.metadata['ZeroPoint'][0]

    @property
    def Mag0(self) -> np.float64:                         # Magnitude 0
        return self.metadata['Mag0'][0]

    @property
    def zero_point_type(self) -> str:                     # Zero point type ('Pogson', 'Asinh', 'Linear', see https://www.ivoa.net/documents/Notes/SVOFPS/NOTE-SVOFPS-1.0.20121015.pdf)
        return self.metadata['ZeroPointType'][0]

    @property
    def softening_parameter(self) -> str:                 # Softening parameter (see https://www.ivoa.net/documents/Notes/SVOFPS/NOTE-SVOFPS-1.0.20121015.pdf)
        return self.metadata['AsinhSoft'][0]

    @property
    def folder(self) -> os.PathLike:                      # Folder of the filter
        folder = Path(FilterPath.Path.path / self.facility / self.instrument)
        if not(os.path.exists(folder)):
            self._logger.debug(f"<Folder {folder} does not exist. Creating it.>")
            folder.mkdir(exist_ok=True, parents=True)
        return folder

    @property
    def filter_path(self) -> os.PathLike:                 # Path of the filter
        return Path(str(self.folder / self.filter_id) + '.fits')

    # # # # # # # # # # # # # # # # # # # # # # # #
    # Properties related to wavelength (see https://www.ivoa.net/documents/Notes/SVOFPS/NOTE-SVOFPS-1.0.20121015.pdf for more details)
    # # # # # # # # # # # # # # # # # # # # # # # #

    @property
    def mean_wavelength(self) -> u.quantity.Quantity:     # Mean integrated wavelength ($\lambda_{mean} = \frac{int_{\lambda} \lambda T(\lambda) d\lambda}{T(\lambda) d\lambda}$)
        return (self.metadata['WavelengthMean'][0] * self.native_unit).to(self.unit)

    @property
    def central_wavelength(self) -> u.quantity.Quantity:  # Central wavelength between the 2 wavelengths used to compute the FWHM
        return (self.metadata['WavelengthCen'][0] * self.native_unit).to(self.unit)

    @property
    def wavelength_eff(self) -> u.quantity.Quantity:      # Mean integrated wavelength with Vega spectrum ($\lambda_{ref} = \frac{\int_{\lambda} \lambda T(\lambda) Vega(\lambda) d\lambda}{T(\lambda) Vega(\lambda) d\lambda}$)
        return (self.metadata['WavelengthEff'][0] * self.native_unit).to(self.unit)

    @property
    def peak_wavelength(self) -> u.quantity.Quantity:     # Wavelength corresponding to the maximum of transmission
        return (self.metadata['WavelengthPeak'][0] * self.native_unit).to(self.unit)

    @property
    def pivot_wavelength(self) -> u.quantity.Quantity:    # Wavelength computed as \sqrt{\frac{\lambda T(\lambda) d\lambda}{T(\lambda) d\lambda / \lambda}}
        return (self.metadata['WavelengthPivot'][0] * self.native_unit).to(self.unit)

    @property
    def photon_wavelength(self) -> u.quantity.Quantity:   # Photon distribution based effective wavelength ($\lambda_{phot} = \frac{\int_{\lambda} \lambda^2 T(\lambda) Vega(\lambda) d\lambda}{\lambda T(\lambda) Vega(\lambda) d\lambda}$)
        return (self.metadata['WavelengthPhot'][0] * self.native_unit).to(self.unit)

    @property
    def wavelength_min(self) -> u.quantity.Quantity:      # Minimum wavelength with transmission > 1% of maximum transmission
        return (self.metadata['WavelengthMin'][0] * self.native_unit).to(self.unit)

    @property
    def wavelength_max(self) -> u.quantity.Quantity:      # Maximum wavelength with transmission > 1% of maximum transmission
        return (self.metadata['WavelengthMax'][0] * self.native_unit).to(self.unit)

    @property
    def effective_width(self) -> u.quantity.Quantity:     # Equivalent to the width of a rectangle with height equal to maximum transmission and with the same area that the one covered by the filter transmission curve ($Width_{eff} = \frac{T(\lambda) d\lambda}{Max(T(\lambda))}$)
        return (self.metadata['WidthEff'][0] * self.native_unit).to(self.unit)


    #################################################
    # Methods
    #################################################

    def _svo_filter_trans(self):
        '''
        Method to query the filter directly online.

        Authors: Allan Denis
        '''
        try:
            self._logger.debug(f"<Look for the filter in the folder {self.folder}>")
            self._load_filter_data_from_fits()
        except ForMoSAError as e:
            try:
                self._logger.info(f"<Recovering filter data in the folder {self.folder} produced the following error: {e}>")
                self._logger.debug(f"<Query the filter {self.name} in the Spanish Virtual Observatory's Filter Profile Service>")
                self._load_filter_data_from_svo()

            except IndexError:
                self._logger.error(f"<No filter found for requested SVO Filter {self.name}>")
                raise ForMoSAError(f"<No filter found for requested SVO Filter {self.name}>")

        except Exception as e:
            self._logger.error(e)
            raise ForMoSAError(e)


    def _plot_transmission_curve(self):
        '''
        Method to plot the transmission curve

        Authors: Allan Denis
        '''

        self._logger.info(f"<Transmission curve of filter {self.name}>")
        plt.figure()
        plt.plot(self.wavelength, self.transmission)
        plt.xlabel(f'Wavelength ({self.unit})')
        plt.ylabel('Transmission Fraction')
        plt.title('Filter Curve for ' + self.name)


    def _set_unit(self, unit: WavelengthUnit):
        '''
        Method to set the unit used for the wavelength

        Parameters
        ----------
        unit (WavelengthUnit): unit used (micrometer', 'nanometer', 'angstrom')

        Authors: Allan Denis
        '''
        self._logger.debug(f"<Convert the unit used for {self.name} from {self.native_unit} to {unit.unit}>")
        if not isinstance(unit, WavelengthUnit):
            self._logger.error(f"<Unit must be a WavelengthUnit Enum, got {type(unit)}>")
            raise ForMoSAError(f"<Unit must be a WavelengthUnit Enum, got {type(unit)}>")

        self._display_unit = unit
        self._logger.info(f"<Unit of wavelength is now in {unit.unit}>")


    def _save_filter_to_path(self, path: str | os.PathLike = None):
        """
        Save filter data and metadata into a FITS file.

        Parameters
        ----------
        path (str | os.PathLike): Output FITS file path. if no path is specified, the method save the data to self.path
        """

        if path is None:
            path = self.filter_path
        else:
            path = Path(path)
            self._logger.debug(f"<Creating directory {path}>")
            path.mkdir(parents=True, exist_ok=True)

        # ==========================
        # PRIMARY HDU
        # ==========================
        primary_hdu = fits.PrimaryHDU()
        hdr = primary_hdu.header

        hdr['FACILITY'] = self.facility
        hdr['INSTR'] = self.instrument
        hdr['FILTERID'] = self.filter_id
        hdr['WAVUNIT'] = self.native_unit.to_string()
        hdr['ORIGIN'] = 'SVO FPS'

        # ==========================
        # TRANSMISSION TABLE
        # ==========================
        wavelength = self.data['Wavelength']
        transmission = self.data['Transmission']

        cols = [
            fits.Column(
                name='Wavelength',
                array=wavelength,
                format='D',
                unit=self.native_unit.to_string()
            ),
            fits.Column(
                name='Transmission',
                array=transmission,
                format='D'
            )
        ]

        trans_hdu = fits.BinTableHDU.from_columns(
            cols,
            name='TRANSMISSION'
        )

        # ==========================
        # METADATA TABLE
        # ==========================
        new_meta = Table()

        for col in self.metadata.colnames:
            data = self.metadata[col]

            if data.dtype.kind in ("U", "S", "O"):
                # Déterminer longueur max
                maxlen = max(len(str(v)) for v in data if v is not None)
                new_meta[col] = Column(
                    np.array(data, dtype=f'U{maxlen}')
                )
            else:
                new_meta[col] = data


        meta_hdu = fits.BinTableHDU(
            new_meta,
            name='METADATA'
        )

        # ==========================
        # WRITE FILE
        # ==========================
        hdul = fits.HDUList([primary_hdu, trans_hdu, meta_hdu])
        hdul.writeto(path, overwrite=True)

        self._logger.info(f"<Filter {self.name} saved to path {path}>")


    def _load_filter_data_from_svo(self):
        '''
        Method to Query filter data in the Spanish Virtual Observatory's Filter Profile Service

        Authors: Mickael Bonnefoy and Allan Denis
        '''

        data = SvoFps.get_transmission_data(self.name)
        metadata = SvoFps.get_filter_list(facility=self.facility, instrument=self.instrument)
        metadata = metadata[metadata['filterID'] == self.name]

        SvoFps.clear_cache()

        self._data = data
        self._metadata = metadata

        self._logger.info(f"<Filter data for {self.name} successfully found>")


    def _load_filter_data_from_fits(self):
        """
        Method to load filter data from the fits file.

        Authors: Allan Denis
        """

        # Convert to Path if it's a string
        fits_path = self.filter_path

        if not fits_path.exists():
            self._logger.error(f"<File {fits_path} does not exist.>")
            raise ForMoSAError(f"<File {fits_path} does not exist.>")

        # Open the FITS file
        self._logger.debug(f"<Open fits file {fits_path}>")
        with fits.open(fits_path) as hdul:

            # ==========================
            # PRIMARY HDU
            # ==========================
            primary_hdu = hdul[0]
            primary_header = primary_hdu.header

            # Extract header information
            self._facility = primary_header.get('FACILITY')
            self._instrument = primary_header.get('INSTR')
            self._filter_id = primary_header.get('FILTERID')
            Unit = primary_header.get('WAVUNIT')
            self._native_unit = WavelengthUnit[Unit.lower()]

            # ==========================
            # TRANSMISSION TABLE (HDU 1)
            # ==========================
            trans_hdu = hdul[1]
            transmission_data = trans_hdu.data

            wavelength, transmission = transmission_data['Wavelength'], transmission_data['Transmission']
            # Mask transmission values if necessary (for example, to mask invalid values like NaN or zeros)
            wavelength_column = MaskedColumn(wavelength, name='Wavelength', unit=self.native_unit, length=len(wavelength))
            transmission_column = MaskedColumn(transmission, name='Transmission', unit='', length=len(transmission))

            transmission_data = Table([wavelength_column, transmission_column])

            # Extract wavelength and transmission data
            self._data = transmission_data

            # ==========================
            # METADATA TABLE (HDU 2)
            # ==========================
            meta_hdu = hdul[2]
            self._metadata = Table(meta_hdu.data)

            self._logger.info(f"<Filter data for {self.name} successfully found>")