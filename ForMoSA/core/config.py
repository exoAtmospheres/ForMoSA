import os
from pathlib import Path
from dataclasses import dataclass, field
from matplotlib import colors as mcolors

from ForMoSA.core.errors import ForMoSAError

# Default filter path (can be overridden by user)
FILTER_PATH = Path.home() / 'ForMoSA' / 'filters'

def set_filter_path(path: str | os.PathLike) -> None:
    '''
    Set the global path used to store photometry filters.

    Parameters
    ----------
    path : str | os.PathLike, Path to the filters folder

    Authors: Allan Denis
    '''

    global FILTER_PATH
    FILTER_PATH = Path(path).expanduser().resolve()

def darken_color(color: str, factor: float = 0.7) -> str:
    '''
    Darken a matplotlib color.

    Parameters
    ----------
    color    (str): Any matplotlib-compatible color.
    factor (float): Multiplicative factor (<1 darker, >1 lighter)

    Returns
    -------
    str : hex color

    Authors: Allan Denis
    '''

    rgb = mcolors.to_rgb(color)
    dark_rgb = tuple(max(0, min(1, c * factor)) for c in rgb)
    return mcolors.to_hex(dark_rgb)

# ==================================================
# Spectroscopic plotting configuration
# ==================================================

@dataclass
class SpectralPlotConfig:
    '''
    Dataclass to handle configurations for plotting spectroscopic data.

    Authors: Allan Denis
    '''

    figsize: tuple[float, float] = (10.0, 7.0)
    label: bool = True
    color: str = "blue"
    edgecolor: str = None
    marker: str = "s"
    markersize: int = 40
    linewidth: float = 1.0
    errorbar_fmt: str = "None"
    errorbar_alpha: float = 1.0
    errorbar_capsize: float = 4.0
    zorder_data: int = 3
    zorder_error: int = 1

    def __post_init__(self):
        if self.edgecolor is None or self.edgecolor == "auto":
            self.edgecolor = darken_color(self.color)

    def to_dict(self) -> dict:
        '''
        Return configuration options as a dictionary.

        Authors: Allan Denis
        '''

        return {k: v for k, v in self.__dict__.items() if v is not None}

    def set_spectral_plot_config(self, **kwargs) -> None:
        '''
        Update global default plotting parameters for spectroscopic plots.

        Parameters
        ----------
        **kwargs : dict, Keyword arguments to override attributes of the config

        Authors: Allan Denis
        '''

        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ForMoSAError(f'<Unknown spectral plot config key: {key}>')
            setattr(self, key, value)

        self.edgecolor = darken_color(self.color)

SPECTRAL_PLOT = SpectralPlotConfig()

# ==================================================
# Photometric plotting configuration
# ==================================================

@dataclass
class PhotometricPlotConfig:
    '''
    Dataclass to handle configurations for plotting photometric data.

    Authors: Allan Denis
    '''
    figsize: tuple[float, float] = (10.0, 7.0)
    label: bool = True
    color: str = "red"
    edgecolor: str = None
    marker: str = "s"
    markersize: int = 70
    linewidth: float = 2.0
    errorbar_fmt: str = "None"
    errorbar_alpha: float = 1.0
    errorbar_capsize: float = 4.0
    zorder_data: int = 3
    zorder_error: int = 1

    def __post_init__(self):
        if self.edgecolor is None or self.edgecolor == "auto":
            self.edgecolor = darken_color(self.color)

    def to_dict(self) -> dict:
        '''
        Return configuration options as a dictionary.

        Authors: Allan Denis
        '''

        return {k: v for k, v in self.__dict__.items() if v is not None}

    def set_photometric_plot_config(self, **kwargs) -> None:
        '''
        Update global default plotting parameters for photometric plots.

        Parameters
        ----------
        **kwargs : dict, Keyword arguments to override attributes of the config

        Authors: Allan Denis
        '''

        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ForMoSAError(f'<Unknown photometric plot config key: {key}>')
            setattr(self, key, value)

        self.edgecolor = darken_color(self.color)

PHOTOMETRIC_PLOT = PhotometricPlotConfig()

# ==================================================
# CornerPlot plotting configuration
# ==================================================

@dataclass
class CornerPlotConfig:
    '''
    Dataclass to handle configurations for corner plots.

    Authors: Allan Denis
    '''

    figsize: tuple[float, float] = (15.0, 15.0)
    bins: int = 30
    smooth: float | None = None
    smooth1d: float | None = None
    plot_datapoints: bool = True
    plot_density: bool = True
    plot_contours: bool = True
    quantiles: tuple = (0.16, 0.5, 0.84)
    show_titles: bool = True
    title_fmt: str = ".3f"
    title_kwargs: dict | None = None
    label_kwargs: dict | None = None
    hist_kwargs: dict | None = None
    contour_kwargs: dict | None = None
    color: str = "magenta"
    max_n_ticks: int = 4

    @property
    def to_dict(self) -> dict:
        '''
        Return configuration options as a dictionary.

        Authors: Allan Denis
        '''

        return {k: v for k, v in self.__dict__.items() if v is not None}

    def set_corner_plot_config(self, **kwargs) -> None:
        '''
        Update global default plotting parameters for corner plot.

        Parameters
        ----------
        **kwargs : dict, Keyword arguments to override attributes of the config

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

    figsize: tuple[float, float] = (15.0, 15.0)

    color_chains: str = 'magenta'
    alpha_chains: float = 0.8

    color_plot_burn_in: str = 'red'
    fontsize_burn_in: int = 14
    text_burn_in: tuple[float, float] = 0.8, 0.8
    color_text_burn_in: str = 'red'
    linestyle_burn_in: str = '--'

    show_weights: bool = True
    color_plot_weights: str = 'black'
    fontsize_weights: int = 14
    alpha_weights: float = 0.4
    text_weights: tuple[float, float] = 0.8, 0.7
    color_text_weights: str = 'grey'

    plot_best_value: bool = True
    color_best_value: str = 'black'
    linestyle_best_value: str = '--'

    @property
    def to_dict(self) -> dict:
        '''
        Return configuration options as a dictionary.

        Authors: Allan Denis
        '''

        return {k: v for k, v in self.__dict__.items() if v is not None}

    def set_chains_plot_config(self, **kwargs) -> None:
        '''
        Update global default plotting parameters for chains plot.

        Parameters
        ----------
        **kwargs : dict, Keyword arguments to override attributes of the config

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

    figsize: tuple[float, float] = (6.0, 6.0)

    color_radar: str = 'magenta'
    linewidth: float = 2.0

    fontsize_names: int = 12

    fontisze_ticks: int = 8
    color_ticks: str = 'black'

    alpha_fill: float = 0.2
    quantiles: tuple[float, float] = (0.16, 0.84)
    size_quantiles: int = 20
    color_quantiles: str = 'black'

    @property
    def to_dict(self) -> dict:
        '''
        Return configuration options as a dictionary.

        Authors: Allan Denis
        '''

        return {k: v for k, v in self.__dict__.items() if v is not None}

    def set_chains_plot_config(self, **kwargs) -> None:
        '''
        Update global default plotting parameters for radar plot.

        Parameters
        ----------
        **kwargs : dict, Keyword arguments to override attributes of the config

        Authors: Allan Denis
        '''

        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ForMoSAError(f'<Unknown radar plot config key: {key}>')
            setattr(self, key, value)

RADAR_PLOT = RadarPlotConfig()

@dataclass
class PlotsConfig:
    CornerPlot: CornerPlotConfig = field(default_factory = lambda: CornerPlotConfig())
    ChainsPlot: ChainsPlotConfig = field(default_factory=lambda: ChainsPlotConfig())
    RadarPlot: RadarPlotConfig = field(default_factory=lambda: RadarPlotConfig())
    SpectralPlot: SpectralPlotConfig = field(default_factory=lambda: SpectralPlotConfig())
    PhotometricPlot: PhotometricPlotConfig = field(default_factory=lambda: PhotometricPlotConfig())

    def __post_init__(self):
        for name, instance in zip(['CornerPlot', 'ChainsPlot', 'RadarPlot', 'SpectralPlot', 'PhotometricPlot'], [CornerPlotConfig, ChainsPlotConfig, RadarPlotConfig, SpectralPlotConfig, PhotometricPlotConfig]):
            value = getattr(self, name)
            if not isinstance(value, instance):
                raise ForMoSAError(f'Wrong type for {name}. Expected {instance}')

    @property
    def to_dict(self) -> dict:
        data = {}

        for name in ['CornerPlot', 'ChainsPlot', 'RadarPlot', 'Spectralplot', 'PhotometricPlot']:
            value = getattr(self, name).to_dict
            data[name] = value

        return data