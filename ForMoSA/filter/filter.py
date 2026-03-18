import os
import logging
import astropy
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from astroquery.svo_fps import SvoFps
from matplotlib.axes._axes import Axes

from pathlib import Path
from astropy.io import fits
from ForMoSA.core.config import FILTER_PATH
from ForMoSA.core.errors import ForMoSAError
from ForMoSA.core.enums import WavelengthUnit
from ForMoSA.core.loggings import setup_logging
from astropy.table import Table, MaskedColumn, Column
from ForMoSA.core.config import  MainPlotConfig, MAIN_PLOT, PhotometricPlotConfig

class PhotometryFilter(object):
    '''
    PhotometryFilter class Defining a filter.
    The faclity, instrument and filter_id has to be under the right format (see https://svo2.cab.inta-csic.es/theory/fps/ for more details).

    Parameters
    ----------
    facility : str
        Name of the facility ('Paranal', 'Keck', 'JWST', ...)
    instrument : str
        Name of the instrument ('SPHERE', 'NIRC2', 'NIRCam', ...)
    filter_id : str
        ID of the filter ('IRDIS_B_H', 'Lp', F410M)
    filter_path : str | os.PathLike
        Path where to save the filter data
    log_level : str
        Level of the logger
    logger : logging.Logger
        Logger
    display_unit : WavelengthUnit
        Unit of the wavelength to display

    Notes
    -----
    Authors: Allan Denis
    '''

    def __init__(self, facility: str, instrument: str, filter_id: str, filter_path: str | os.PathLike | None = None, log_level: str = 'info', logger: logging.Logger | None = None, display_unit: WavelengthUnit = WavelengthUnit.MICROMETER):
        if logger == None:
            self._logger = setup_logging(level=log_level, name=__name__)
        else:
            self._logger = logger

        self._facility = facility
        self._instrument = instrument
        self._filter_id = filter_id
        self._filter_path = filter_path if filter_path is not None else FILTER_PATH

        self._data = []
        self._medata = []

        self._native_unit = WavelengthUnit.ANGSTROM
        self._display_unit = display_unit

        self._validate()
        self._svo_filter_trans()

    # ================================================
    # Representation
    # ================================================

    def __repr__(self) -> str:
        return f'<PhotometryFilter, name={self.name}, central wavelength = {self.central_wavelength}, range = [{self.wavelength_min} - {self.wavelength_max}] {self.unit}>'

    def __format__(self) -> str:
        return self.__repr__()

    # ===============================================
    # Properties
    # ===============================================

    # ------------------
    # General properties
    # ------------------

    @property
    def logger(self) -> logging.Logger:
        """Logger."""
        return self._logger

    @property
    def facility(self) -> str:
        """Facility for the filter (e.g. 'Paranal', 'Keck', 'JWST', ...)."""
        return self._facility

    @property
    def instrument(self) -> str:
        """Instrument used of the filter (e.g. 'SPHERE', 'NIRC2', 'NIRCam', ...)."""
        return self._instrument

    @property
    def filter_id(self) -> str:
        """ID of the filter (e.g. 'IRDIS_B_H', 'Lp', 'F410M', ...)."""
        return self._filter_id

    @property
    def filter_path(self) -> Path:
        return Path(self._filter_path)

    @filter_path.setter
    def filter_path(self, new_path: str | os.PathLike) -> None:
        self._filter_path = new_path

    @property
    def name(self) -> str:
        """Full name of the filter (e.g. 'Paranal/SPHERE.IRDIS_B_H', 'Keck/NIRC2.Lp', 'JWST/NIRCam.F410M')."""
        return self.facility + '/' + self.instrument + '.' + self.filter_id

    @property
    def data(self) -> astropy.table.table.Table:
        """Table containing the data of the filter (wavelength, transmission)."""
        return self._data

    @property
    def unit(self) -> u.core.PrefixUnit:
        """Unit to use for the wavelength."""
        return self._display_unit.unit

    @property
    def native_unit(self) -> u.core.PrefixUnit:
        """Native unit of the wavelength (Angstrom)."""
        return self._native_unit.unit

    @native_unit.setter
    def native_unit(self, new_unit: WavelengthUnit) -> None:
        """Setter for native unit of the wavelength."""
        self._native_unit = new_unit
        self._validate()

    @property
    def metadata(self) -> astropy.table.table.Table:
        """Table containing the metadata of the filter."""
        return self._metadata

    @property
    def wavelength(self) -> u.quantity.Quantity:
        """Wavelength grid of the filter."""
        return ((self.data['Wavelength'].data.filled(np.nan) * self.native_unit).to(self.unit)).value

    @property
    def transmission(self) -> np.ndarray:
        """Transmission of the filter."""
        return self.data['Transmission'].data.filled(np.nan)

    @property
    def fwhm(self) -> u.quantity.Quantity:
        """FWHM."""
        return ((self.metadata['FWHM'][0] * self.native_unit).to(self.unit)).value

    @property
    def zero_point(self) -> np.float64:
        """Zero point."""
        return self.metadata['ZeroPoint'][0]

    @property
    def Mag0(self) -> np.float64:
        """Magnitude 0."""
        return self.metadata['Mag0'][0]

    @property
    def zero_point_type(self) -> str:
        """Zero point type ('Pogson', 'Asinh', 'Linear', see https://www.ivoa.net/documents/Notes/SVOFPS/NOTE-SVOFPS-1.0.20121015.pdf)."""
        return self.metadata['ZeroPointType'][0]

    @property
    def softening_parameter(self) -> str:
        """Softening parameter (see https://www.ivoa.net/documents/Notes/SVOFPS/NOTE-SVOFPS-1.0.20121015.pdf)."""
        return self.metadata['AsinhSoft'][0]

    @property
    def folder(self) -> Path:
        """Folder of the filter."""
        folder = self.filter_path / self.facility / self.instrument

        if not folder.exists():
            self._logger.info(f"    {folder} does not exist. Creating it")
            folder.mkdir(parents=True, exist_ok=True)

        return folder

    @property
    def file_path(self) -> Path:
        """Path of the filter."""
        return self.folder / f"{self.filter_id}.fits"

    # ---------------------------------------------------------------------------------------------
    # Properties related to wavelength
    # see https://www.ivoa.net/documents/Notes/SVOFPS/NOTE-SVOFPS-1.0.20121015.pdf for more details
    # ---------------------------------------------------------------------------------------------

    @property
    def mean_wavelength(self) -> float:
        r"""Mean integrated wavelength ($\lambda_{mean} = \frac{int_{\lambda} \lambda T(\lambda) d\lambda}{T(\lambda) d\lambda}$)."""
        return ((self.metadata['WavelengthMean'][0] * self.native_unit).to(self.unit)).value

    @property
    def central_wavelength(self) -> float:
        """Central wavelength between the 2 wavelengths used to compute the FWHM."""
        return ((self.metadata['WavelengthCen'][0] * self.native_unit).to(self.unit)).value

    @property
    def wavelength_eff(self) -> float:
        r"""Mean integrated wavelength with Vega spectrum ($\lambda_{ref} = \frac{\int_{\lambda} \lambda T(\lambda) Vega(\lambda) d\lambda}{T(\lambda) Vega(\lambda) d\lambda}$)."""
        return ((self.metadata['WavelengthEff'][0] * self.native_unit).to(self.unit).value)

    @property
    def peak_wavelength(self) -> float:
        """Wavelength corresponding to the maximum of transmission."""
        return ((self.metadata['WavelengthPeak'][0] * self.native_unit).to(self.unit)).value

    @property
    def pivot_wavelength(self) -> float:
        r"""Wavelength computed as \sqrt{\frac{\lambda T(\lambda) d\lambda}{T(\lambda) d\lambda / \lambda}}."""
        return ((self.metadata['WavelengthPivot'][0] * self.native_unit).to(self.unit)).value

    @property
    def photon_wavelength(self) -> float:
        r"""Photon distribution based effective wavelength ($\lambda_{phot} = \frac{\int_{\lambda} \lambda^2 T(\lambda) Vega(\lambda) d\lambda}{\lambda T(\lambda) Vega(\lambda) d\lambda}$)."""
        return ((self.metadata['WavelengthPhot'][0] * self.native_unit).to(self.unit)).value

    @property
    def wavelength_min(self) -> float:
        """Minimum wavelength with transmission > 1% of maximum transmission."""
        return ((self.metadata['WavelengthMin'][0] * self.native_unit).to(self.unit)).value

    @property
    def wavelength_max(self) -> float:
        """Maximum wavelength with transmission > 1% of maximum transmission."""
        return ((self.metadata['WavelengthMax'][0] * self.native_unit).to(self.unit)).value

    @property
    def wavelength_range(self) -> tuple[float, float]:
        """Wavelength range."""
        return self.wavelength_min, self.wavelength_max

    @property
    def effective_width(self) -> float:
        r"""Equivalent to the width of a rectangle with height equal to maximum transmission and with the same area that the one covered by the filter transmission curve ($Width_{eff} = \frac{T(\lambda) d\lambda}{Max(T(\lambda))}$)."""
        return ((self.metadata['WidthEff'][0] * self.native_unit).to(self.unit)).value

    @property
    def width(self) -> np.ndarray[float, float]:
        """Width of the filter."""
        return np.array([self.central_wavelength - self.wavelength_min, self.wavelength_max - self.central_wavelength])[:,np.newaxis]

    # ===============================================
    # Class method
    # ===============================================

    @classmethod
    def _from_filter_name(cls, filter_name: str, filter_path: str | os.PathLike | None = None, logger: logging.Logger | None = None, log_level: str = 'INFO', display_unit: WavelengthUnit = WavelengthUnit.MICROMETER) -> 'PhotometryFilter':
        '''
        Build an instance of Filter from the filter_name.
        filter_name must be under the format facility/instrument.filter_id (e.g. 'Keck/NIRC2.Lp')

        Examples
        --------
        >>> Filter = PhotometryFilter.from_filter_name('Keck/NIRC2.Lp')

        Parameters
        ----------
        filter_name : str
            Full name of the filter

        Returns
        -------
        'PhotometryFilter'
            Instance of PhotometryFilter

        Notes
        -----
        Authors: Allan Denis
        '''

        facility, instrument, filter_id = filter_name.split('/')[0], filter_name.split('/')[1].split('.')[0], filter_name.split('/')[1].split('.')[1]

        return cls(facility=facility, instrument=instrument, filter_id=filter_id, filter_path=filter_path, logger=logger, log_level=log_level, display_unit=display_unit)

    # ===============================================
    # Methods
    # ===============================================

    def _validate(self) -> None:
        '''
        Validate the filter data.

        Notes
        -----
        Authors: Allan Denis
        '''

        if not isinstance(self._filter_path, (str, os.PathLike)):
            raise ForMoSAError(f"<Wrong type for filter_path: {type(self._filter_path)}. Expected str or os.PathLike>", self.logger)

        if not isinstance(self._facility, str):
            raise ForMoSAError(f"<Wrong type for facility: {type(self._facility)}. Expected str>", self.logger)

        if not isinstance(self._instrument, str):
            raise ForMoSAError(f"<Wrong type for instrument: {type(self._instrument)}. Expected str>", self.logger)

        if not isinstance(self._filter_id, str):
            raise ForMoSAError(f"<Wrong type for filter_id: {type(self._filter_id)}. Expected str>", self.logger)

        if not isinstance(self._native_unit, WavelengthUnit):
            raise ForMoSAError(f"<Wrong type for native_unit: {type(self._native_unit)}. Expected a WavelengthUnit>", self.logger)

        if not isinstance(self._display_unit, WavelengthUnit):
            raise ForMoSAError(f"<Wrong type for display_unit: {type(self._display_unit)}. Expected a WavelengthUnit>", self.logger)

    def _svo_filter_trans(self) -> None:
        '''
        Get the filter data either from a local FITS file or from the Spanish Virtual Observatory's Filter Profile Service.

        Notes
        -----
        Authors: Allan Denis
        '''

        try:
            self._logger.debug(f"<Look for the filter {self.name} in the folder {self.folder}>")
            self._load_filter_data_from_fits()
        except ForMoSAError as e:
            try:
                self._logger.info(f"<Recovering filter {self.name} in the folder {self.folder} produced the following error: {e}. Looking in the Spanish Virtual Observation's FIlter Profile Service>")
                self._load_filter_data_from_svo()
                self._logger.debug(f"<Saving the filter {self.name} to the folder {self.folder}>")
                self._save_filter()

            except IndexError:
                raise ForMoSAError(f"<No filter found for requested SVO Filter {self.name}>", self.logger)

        except Exception as e:
            raise ForMoSAError(e, self.logger)

    def _plot_transmission_curve(self, fig: Figure | None = None, ax: Axes | None = None, plot_config: PhotometricPlotConfig = PhotometricPlotConfig(), main_plot_config: MainPlotConfig = MAIN_PLOT) -> None:
        '''
        Method to plot the transmission curve

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            Figure (used to overplot on an existing figure)
        ax : matplotlib.axes._axes.Axes
            Ax (used to overplot on an existing ax)
        plot_config : PhotometricPlotConfig
            Instance of class PhotometricPlotConfig
        main_plot_config : MainPlotConfig
            Instance of class MainPlotConfig

        Returns
        -------
        fig : matplotlib.figure.Figure
            Updated figure
        ax : matplotlib.axes._axes.Axes
            Updated ax

        Notes
        -----
        Authors: Allan Denis
        '''

        self._logger.info(f"      Plotting transmission curve of filter {self.name}")

        # --------------------------------------------------
        # Figure / axes creation
        # --------------------------------------------------
        if ax is None:
            fig, ax = plt.subplots(figsize=main_plot_config.figsize)
            ax.set_xlabel(f'Wavelength ({self.unit})')
            ax.set_ylabel('Transmission')

        if plot_config.label_filter:
            ax.plot(self.wavelength, self.transmission, color = plot_config.color, label = f'{self.name}')
            ax.legend(fontsize=plot_config.legend_fontsize)
        else:
            ax.plot(self.wavelength, self.transmission, color = plot_config.color)

        return fig, ax

    def _set_unit(self, unit: WavelengthUnit) -> None:
        '''
        Method to set the unit used for the wavelength

        Parameters
        ----------
        unit : WavelengthUnit
            unit used (micrometer', 'nanometer', 'angstrom')

        Notes
        -----
        Authors: Allan Denis
        '''

        if not isinstance(unit, WavelengthUnit):
            raise ForMoSAError(f"<Unit must be a WavelengthUnit Enum, got {type(unit)}>", self.logger)
        self._logger.debug(f"<Convert the unit used for Filter {self.name} from {self.native_unit} to {unit.unit}>")


        self._display_unit = unit
        self._validate
        self._logger.info(f"    Setting wavelength unit to {unit.unit} for filter {self.name} to {self.unit}")

    def _save_filter(self) -> None:
        '''
        Save filter data and metadata into a FITS file.

        Notes
        -----
        Authors: Allan Denis
        '''

        path = self.file_path

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

    def _load_filter_data_from_svo(self) -> None:
        '''
        Method to Query filter data in the Spanish Virtual Observatory's Filter Profile Service

        Notes
        -----
        Authors: Mickael Bonnefoy and Allan Denis
        '''

        data = SvoFps.get_transmission_data(self.name)
        metadata = SvoFps.get_filter_list(facility=self.facility, instrument=self.instrument)
        metadata = metadata[metadata['filterID'] == self.name]

        SvoFps.clear_cache()

        self._data = data
        self._metadata = metadata

        self._logger.info(f"    Filter data for {self.name} successfully found")

    def _load_filter_data_from_fits(self) -> None:
        """
        Method to load filter data from the fits file.

        Notes
        -----
        Authors: Allan Denis
        """

        # Convert to Path if it's a string
        fits_path = self.file_path

        if not fits_path.exists():
            raise ForMoSAError(f"File {fits_path} does not exist", self.logger)

        # Open the FITS file
        self._logger.debug(f"Open fits file {fits_path}")
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

            self._logger.info(f"    Filter data for {self.name} successfully found")

