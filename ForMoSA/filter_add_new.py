import numpy as np
import matplotlib.pyplot as plt


def add_filter(pathr, filter_name, plot_filt=False):
    '''
    Function to add filters to the personal phototeque

    
    
    '''
    filter = open(pathr+filter_name+'.txt', 'r')
    x = []
    y = []
    for line in filter:
        if np.logical_or(line[0] == '#', line[0] == '\n'):
            pass
        else:
            line = line.strip().split()
            x.append(float(line[0])/10000)
            y.append(float(line[1]))
    if plot_filt==True:
        plt.plot(x,y)
        plt.show()

    np.savez('phototeque/'+name, x_filt=x, y_filt=y)
