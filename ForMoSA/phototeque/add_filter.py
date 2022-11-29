import numpy as np
import matplotlib.pyplot as plt


# pathr='/Users/petruss/Documents/ForMoSA_3.0/Phototeque/SPHERE_K2.txt'
pathr='/Users/simonpetrus/Downloads/WISE_WISE.W2.dat'
filer = open(pathr, 'r')
x = []
y = []
for line in filer:
    if np.logical_or(line[0] == '#', line[0] == '\n'):
        pass
    else:
        line = line.strip().split()
        x.append(float(line[0])/10000)
        y.append(float(line[1]))

plt.plot(x,y)
plt.show()
np.savez('/Users/simonpetrus/Documents/FORMOSA_DEV/Phototeque/WISE_W2', x_filt=x, y_filt=y)
