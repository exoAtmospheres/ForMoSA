import numpy as np 
import logging
from pathlib import Path
import os 
import xarray as xr

# log
_log = logging.getLogger(__name__)

# Format logging for this module
if not _log.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)  # Minimal logging level for this module
    formatter = logging.Formatter('%(asctime)s\t%(levelname)8s\t%(message)s')
    formatter.default_msec_format = '%s.%03d'
    handler.setFormatter(formatter)
    _log.addHandler(handler)

_log.setLevel(logging.INFO)
_log.propagate = False 


class Model(object):
    '''
    Model class, which provides easy access to a grid of model
    
    Parameters
    ----------
    path : str | os.PathLike
        Path to the grid of model.
    '''
    
    def __init__(self, path: str | os.PathLike) -> None:
        
        # the command .expanduser() transforms the path in a full absolute path, removing any '~' in the path
        self._root = Path(path).expanduser().parent
        self._name = str(Path(path).expanduser()).split('/')[-1].split('.nc')[0]
        
        
    ##################################################
    # Representation
    ##################################################

    def __repr__(self) -> str:
        self._read_info()
        return f'<Model, name={self.name}, root={self.root}, wavelength={min(self.wavelength):.3f}-{max(self.wavelength):.3f}, resolution={min(self.resolution):.3f}-{max(self.resolution):.3f}>'

    def __format__(self) -> str:
        return self.__repr__()

    ##################################################
    # Properties
    ##################################################

    @property  
    def root(self): 
        return self._root

    @property  
    def name(self):
        return self._name 
    
    @property 
    def path(self):
        return Path(str(self.root / self.name) + '.nc')
    
    @property  
    def attrs(self):
        return self._attrs
    
    @property  
    def wavelength(self):
        return self._wavelength
    
    @property  
    def nyquist(self):   # Nyquist resolution
        return self.wavelength / (2 * np.diff(self.wavelength, append=(2*self.wavelength[-1] - self.wavelength[-2])))
    
    @property  
    def resolution(self):   # Resolution defined as the minimum between the given resolution and the Nyquist resolution
        return np.minimum(self.nyquist, self._resolution)
    
    @property  
    def grid(self):
        return self._grid

    
    ##################################################
    # Methods
    ##################################################
    
    def _read_info(self):
        '''
        Read the model grid and store important information
        '''
        
        _log.info('Read model grid')
        ds = xr.open_dataset(self.path, decode_cf=False, engine="netcdf4")
        self._wavelength = ds['wavelength'].values
        self._resolution = ds.attrs['res']
        self._attrs = ds.attrs
        self._attrs['res'] = self.resolution
        self._grid = ds['grid']
        
        
        
# These lines are just for testing purposes, they will be removed for the final version
model = Model('/Users/allandenis/These/ForMoSA_Main/INPUT_MODELS/EXOREM_native.nc')
model._read_info()