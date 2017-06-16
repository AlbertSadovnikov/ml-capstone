import numpy as np
import matplotlib.pyplot as plt

data = np.load('transforms00.npz')

t = data['transforms']
rt = t.reshape((-1, 9))
f, ax = plt.subplots(3, 3)
for i in range(3):
    for j in range(3):
        ax[i, j].plot(t[:1000, i, j], 'b-')
plt.show()

