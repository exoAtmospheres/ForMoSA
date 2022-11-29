import numpy as np
import matplotlib.pyplot as plt

pathr='/Your/Path/here/'
name = 'Filter_Name'
filter = open(pathr+name+'.txt', 'r')
x = []
y = []
for line in filter:
    if np.logical_or(line[0] == '#', line[0] == '\n'):
        pass
    else:
        line = line.strip().split()
        x.append(float(line[0])/10000)
        y.append(float(line[1]))

plt.plot(x,y)
plt.show()
np.savez('phototeque/'+name, x_filt=x, y_filt=y)
