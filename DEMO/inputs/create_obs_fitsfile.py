import pandas as pd
from astropy.table import Table

def creat_fitsfile(name_file):
    '''
    Works from old format (txt file with: wave, flux, flux_err, flag, type, resolution, instrumen) to new fits format
    '''
    fileop = pd.read_csv(name_file+'.txt', header=None, sep='\s+')
    fileop.columns=['WAV', 'FLX', 'ERR', 'a', 'b', 'RES', 'INS']
    fileop = fileop.drop(['a', 'b'], axis=1)

    t = Table.from_pandas(fileop)
    t.write(name_file+'.fits')