import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

d0 = np.load('transforms00.npz')
d1 = np.load('transforms02.npz')

t0 = d0['transforms']
t1 = d1['transforms']
f, ax = plt.subplots(3, 3)
for i in range(3):
    for j in range(3):
        x0 = t0[:1000, i, j]
        x1 = t1[:1000, i, j]
        ax[i, j].plot(x0, 'b-')
        ax[i, j].plot(x1, 'r-')
plt.show()

