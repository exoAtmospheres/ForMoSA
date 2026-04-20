import os
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from matplotlib import colors as mcolors

from ForMoSA.core.errors import ForMoSAError

# Default filter path (can be overridden by user)
FILTER_PATH = Path.home() / 'filters'

def set_filter_path(path: str | os.PathLike) -> None:
    '''
    Set the global path used to store photometry filters.

    Parameters
    ----------
    path : str | os.PathLike
        Path to the filters folder

    Notes
    -----
    Authors: Allan Denis
    '''

    global FILTER_PATH
    FILTER_PATH = Path(path).expanduser().resolve()

def darken_color(color: str, factor: float = 0.7) -> str:
    '''
    Darken a matplotlib color.

    Parameters
    ----------
    color : str
        Any matplotlib-compatible color.
    factor : float
        Multiplicative factor (<1 darker, >1 lighter)

    Returns
    -------
    str : hex color

    Notes
    -----
    Authors: Allan Denis
    '''

    rgb = mcolors.to_rgb(color)
    dark_rgb = tuple(max(0, min(1, c * factor)) for c in rgb)
    return mcolors.to_hex(dark_rgb)

# ==================================================
# General plotting configuration
# ==================================================

@dataclass
class MainPlotConfig:
    '''
    Dataclass to handle main parameters for plotting.

    Notes
    -----
    Authors: Allan Denis
    '''

    figsize: tuple[float, float] = (10.0, 7.0)
    legend_ncol: int = 1
    legend_hc_ncol: int = 1
    legend_filt_ncol: int = 1
    legend_fontsize: str = "small"
    minor_ticks: bool = True
    nb_minor_ticks: int = 5

MAIN_PLOT = MainPlotConfig()

# ==================================================
# Spectroscopic plotting configuration
# ==================================================

@dataclass
class ObsPlotConfig:
    '''
    Dataclass to handle configurations for plotting data.

    Notes
    -----
    Authors: Allan Denis
    '''


    # --- Color management
    cmap: mcolors.Colormap = field(default_factory = lambda: cm.jet)
    color: str = "#7A1E22"
    edgecolor: str = None
    norm: mcolors.Normalize = field(default_factory = lambda: plt.Normalize(vmin=1.0, vmax=15))

    # --- Marker
    marker: str = "s"
    markersize: int = 40
    linewidth: float = 1

    # --- Errorbar
    errorbar_fmt: str = "None"
    errorbar_alpha: float = 1.0
    errorbar_capsize: float = 4.0

    # --- zorder
    zorder_data: int = 3
    zorder_error: int = 1

    # ---- Legend
    label: bool = True

    def __post_init__(self):
        if self.edgecolor is None or self.edgecolor == "auto":
            self.edgecolor = darken_color(self.color)

    def to_dict(self) -> dict:
        '''
        Return configuration options as a dictionary.

        Notes
        -----
        Authors: Allan Denis
        '''

        return {k: v for k, v in self.__dict__.items() if v is not None}

    def set_plot_config(self, **kwargs) -> None:
        '''
        Update global default plotting parameters for spectroscopic plots.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to override attributes of the config

        Notes
        -----
        Authors: Allan Denis
        '''

        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ForMoSAError(f'Unknown spectral plot config key: {key}')
            setattr(self, key, value)

        self.edgecolor = darken_color(self.color)

@dataclass
class SpectralPlotConfig(ObsPlotConfig):
    '''
    Dataclass to handle configurations for plotting spectroscopic data.

    Notes
    -----
    Authors: Allan Denis
    '''

    color: str = "#7A1E22"
    marker: str = "None"

# ==================================================
# Photometric plotting configuration
# ==================================================

@dataclass
class PhotometricPlotConfig(ObsPlotConfig):
    '''
    Dataclass to handle configurations for plotting photometric data.

    Notes
    -----
    Authors: Allan Denis
    '''

    color: str = "blue"
    markersize: int = 70
    linewidth: float = 2.0

    label_filter: bool = False
    label_data: bool = True

# ==================================================
# CornerPlot plotting configuration
# ==================================================

@dataclass
class CornerPlotConfig:
    '''
    Dataclass to handle configurations for corner plots.

    Notes
    -----
    Authors: Allan Denis
    '''

    figsize: tuple[float, float] = (15.0, 15.0)
    color: str = '#A12A1F'
    bins: int = 80
    smooth: float = 1
    smooth1d: float | None = None
    plot_datapoints: bool = False
    plot_density: bool = True
    plot_contours: bool = True
    fill_contours: bool = True
    quantiles: tuple = (0.16, 0.5, 0.84)
    # levels: list = field(default_factory = lambda: [0.997, 0.95, 0.68]) # 3-sigma, 2-sigma, 1-sigma but for 1D Gaussian
    levels: list = field(default_factory = lambda: [0.3935, 0.8647, 0.9889]) # 1-sigma, 2-sigma, 2-sigma for 2D Gaussian
    show_titles: bool = True
    title_fmt: str = " .2f"
    hist_kwargs: dict = field(default_factory = lambda: dict(color='#A12A1F', histtype='stepfilled', alpha=0.6, edgecolor='#5B1218', linewidth=0.8))
    # contour_kwargs: dict | None = None
    contour_kwargs: dict = field(default_factory = lambda: dict(colors='#A12A1F', linewidths=0.8))
    pcolor_kwargs: dict = field(default_factory = lambda: dict(color='#5B1218'))
    title_kwargs: dict = field(default_factory = lambda: dict(fontsize=14))
    label_kwargs: dict = field(default_factory= lambda: dict(fontsize=14))
    max_n_ticks: int = 4

    @property
    def to_dict(self) -> dict:
        '''
        Return configuration options as a dictionary.

        Notes
        -----
        Authors: Allan Denis
        '''

        return {k: v for k, v in self.__dict__.items() if v is not None}

    def set_corner_plot_config(self, **kwargs) -> None:
        '''
        Update global default plotting parameters for corner plot.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to override attributes of the config

        Notes
        -----
        Authors: Allan Denis
        '''

        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ForMoSAError(f'<Unknown corner plot config key: {key}>')
            setattr(self, key, value)

CORNER_PLOT = CornerPlotConfig()

# ==================================================
# Chains plotting configuration
# ==================================================

@dataclass
class ChainsPlotConfig:
    '''
    Dataclass to handle configurations for chains plot.

    Notes
    -----
    Authors: Allan Denis
    '''

    figsize: tuple[float, float] = (18.0, 12.0)

    color_chains: str = "violet"
    alpha_chains: float = 0.8

    color_plot_burn_in: str = '#A12A1F'
    fontsize_burn_in: int = 14
    text_burn_in: tuple[float, float] = 0.8, 0.8
    color_text_burn_in: str = '#A12A1F'
    linestyle_burn_in: str = '--'

    show_weights: bool = True
    color_plot_weights: str = '#1F1F1F'
    fontsize_weights: int = 14
    alpha_weights: float = 0.4
    text_weights: tuple[float, float] = 0.8, 0.7
    color_text_weights: str = '#1F1F1F'

    plot_best_value: bool = True
    color_best_value: str = 'black'
    linestyle_best_value: str = '-.'

    @property
    def to_dict(self) -> dict:
        '''
        Return configuration options as a dictionary.

        Notes
        -----
        Authors: Allan Denis
        '''

        return {k: v for k, v in self.__dict__.items() if v is not None}

    def set_chains_plot_config(self, **kwargs) -> None:
        '''
        Update global default plotting parameters for chains plot.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to override attributes of the config

        Notes
        -----
        Authors: Allan Denis
        '''

        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ForMoSAError(f'<Unknown chains plot config key: {key}>')
            setattr(self, key, value)

CHAINS_PLOT = ChainsPlotConfig()

# ==================================================
# Radar plotting configuration
# ==================================================

@dataclass
class RadarPlotConfig:
    '''
    Dataclass to handle configurations for radar plot.

    Notes
    -----
    Authors: Allan Denis
    '''

    figsize: tuple[float, float] = (6.0, 6.0)

    # Improved color scheme - using a sophisticated blue-purple gradient
    main_color = '#4A5FD9'  # Deep blue
    fill_color = '#6B7FE8'  # Lighter blue
    uncertainty_color = '#A8B3F5'  # Very light blue

    color_radar: str = '#4A5FD9'
    color_uncertainty: str = '#4A5FD9'  # Use the same deep blue for quantiles
    linewidth: float = 2.0

    fontsize_names: int = 11

    fontisze_ticks: int = 11
    color_ticks: str = '#24292E'

    alpha_fill: float = 0.35
    quantiles: tuple[float, float] = (0.16, 0.84)
    size_quantiles: int = 80
    color_quantiles: str = '#4A5FD9'
    lw_quantiles: float = 2.0

    @property
    def to_dict(self) -> dict:
        '''
        Return configuration options as a dictionary.

        Notes
        -----
        Authors: Allan Denis
        '''

        return {k: v for k, v in self.__dict__.items() if v is not None}

    def set_radar_plot_config(self, **kwargs) -> None:
        '''
        Update global default plotting parameters for radar plot.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to override attributes of the config

        Notes
        -----
        Authors: Allan Denis
        '''

        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ForMoSAError(f'<Unknown radar plot config key: {key}>')
            setattr(self, key, value)

RADAR_PLOT = RadarPlotConfig()

# ==================================================
# Best fit plotting configuration
# ==================================================

@dataclass
class BestFitPlotConfig:
    '''
    Dataclass to handle configurations for best fit plot.

    Notes
    -----
    Authors: Allan Denis
    '''

    color_fit: str = 'black'
    color_residuals: str = "#2C2C2C"
    linewidth: float = 1.0
    zorder: int = 200

    @property
    def to_dict(self) -> dict:
        '''
        Return configuration options as a dictionary.

        Notes
        -----
        Authors: Allan Denis
        '''

        return {k: v for k, v in self.__dict__.items() if v is not None}

    def set_best_fit_plot_config(self, **kwargs) -> None:
        '''
        Update global default plotting parameters for best fit plot.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to override attributes of the config

        Notes
        -----
        Authors: Allan Denis
        '''

        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ForMoSAError(f'<Unknown best fit plot config key: {key}>')
            setattr(self, key, value)

BEST_FIT_PLOT = BestFitPlotConfig()

@dataclass
class PlotsConfig:
    '''
    Dataclass to handle configurations for plots.

    Notes
    -----
    Authors: Allan Denis
    '''

    CornerPlot: CornerPlotConfig = field(default_factory = lambda: CORNER_PLOT)
    ChainsPlot: ChainsPlotConfig = field(default_factory = lambda: CHAINS_PLOT)
    RadarPlot: RadarPlotConfig = field(default_factory = lambda: RADAR_PLOT)
    BestFitPlot: BestFitPlotConfig = field(default_factory = lambda: BEST_FIT_PLOT)

    def __post_init__(self):
        for name, instance in zip(['CornerPlot', 'ChainsPlot', 'RadarPlot', 'BestFitPlot'], [CornerPlotConfig, ChainsPlotConfig, RadarPlotConfig, BestFitPlotConfig]):
            value = getattr(self, name)
            if not isinstance(value, instance):
                raise ForMoSAError(f'Wrong type for {name}. Expected {instance}')

    @property
    def to_dict(self) -> dict:
        data = {}

        for name in ['CornerPlot', 'ChainsPlot', 'RadarPlot', 'BestFitPlot']:
            value = getattr(self, name).to_dict
            data[name] = value

        return data

PLOTS_CONFIG = PlotsConfig()