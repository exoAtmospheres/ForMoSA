import numpy as np
import matplotlib.pyplot as plt


def add_filter(pathr, filter_name, unit, plot_filt=False):
    '''
    Function to add filters to the personal phototeque
    http://svo2.cab.inta-csic.es/theory/fps/

    
    '''
    filter = open(pathr+filter_name+'.dat', 'r')
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

    np.savez('/home/mravet/Documents/These/FORMOSA/ForMoSA_main/ForMoSA/phototeque/'+filter_name, x_filt=x, y_filt=y)

add_filter('/home/mravet/Documents/These/FORMOSA/INPUTS/DATA/COCONUTS/Photometry/', 'WISE_W1', 'A')

