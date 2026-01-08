__version__ = "1.1.5"

# high-level imports
from .observation import Observation
from .analysis import Analysis
from .global_params import GlobalParameters, GlobalParams
from .nested_sampling.sampling import NestedSampling
from .paths import ForMoSAPaths
from .ForMoSA_logging import setup_logging
from .Filter import Filter
from .ForMoSA_enums import WavelengthUnit, FluxUnit, DataUnit, FilterType

# high-level utility functions
from .phototeque import add_filter, list_filters

