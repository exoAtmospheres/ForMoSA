__version__ = "1.1.5"

# high-level imports
from .observation import Observation
from .analysis import Analysis
from .global_params import GlobalParams

# high-level utility functions
from .phototeque import add_filter, list_filters

# detinition of main ForMoSA error class
class ForMoSAError(Exception):
    pass
