import numpy as np
import matplotlib.pyplot as plt
import os

from pathlib import Path


def add_filter(path_in: str | os.PathLike, filter_name: str, unit: str, path_out: str | os.PathLike, plot_filt: bool = False):
    '''
    Function to add filters to the personal phototeque
    http://svo2.cab.inta-csic.es/theory/fps/
    '''
    path_int = Path(path_in)
    path_out = Path(path_out)

    filter = open(path_int / f'{filter_name}.dat', 'r')
    if unit == 'A':
        conv = 1e4
    if unit == 'micron':
        conv= 1
    if unit == 'nm':
        conv= 1e3
    else:
        print("add the unit of the wavelenght (A, micron, nm)")

    x = [] # units = µm
    y = []
    for line in filter:
        if np.logical_or(line[0] == '#', line[0] == '\n'):
            pass
        else:
            line = line.strip().split()
            x.append(float(line[0])/conv)
            y.append(float(line[1]))
    if plot_filt==True:
        plt.plot(x,y)
        plt.show()

    np.savez(path_out / f'{filter_name}.npz', x_filt=x, y_filt=y)


def list_filters():
    '''
    Function to list the filters in the personal phototeque
    '''
    path = Path(__file__).parent
    filters = [f.stem for f in path.glob('*.npz')]
    filters = sorted(filters)

    return filters


if __name__ == "__main__":
    add_filter('/home/mravet/Documents/These/FORMOSA/INPUTS/DATA/COCONUTS/Photometry/', 'WISE_W1', 'A')

