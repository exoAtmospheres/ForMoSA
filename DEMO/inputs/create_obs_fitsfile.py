import pandas as pd
from astropy.table import Table

def creat_fitsfile(name_file):
    '''
    Easy function to save .fits from dataframe 
    Columns must be: wave [µm], fluxv[W m-2 µm-1], flux_errv[W m-2 µm-1], resolution, instrumen_name
    '''
    fileop = pd.read_csv(name_file+'.txt', header=None, sep='\s+')
    fileop.columns=['WAV', 'FLX', 'ERR', 'RES', 'INS']
    t = Table.from_pandas(fileop)
    t.write(name_file+'.fits')