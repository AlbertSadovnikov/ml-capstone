import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
import cv2

d0 = np.load('transforms03.npz')
t0 = d0['transforms']
rt0 = d0['back_transforms']

n = len(t0)

points = np.array([[0, 0], [0, 1080], [1920, 1080], [1920, 0]], np.float32)

result = np.zeros((n, 4, 2), np.float32)

tform = np.eye(3, dtype=np.float32)

for idx in range(n):
    tform = tform.dot(np.squeeze(t0[idx, :, :]))
    result[idx, :, :] = np.squeeze(cv2.perspectiveTransform(points[None, :, :], tform))

pts_last = np.squeeze(result[-1, :, :])
result_rev = np.zeros((n, 4, 2), np.float32)

tform_rev = np.eye(3, dtype=np.float32)

for idx in range(n):
    tform_rev = tform_rev.dot(np.squeeze(rt0[n - idx - 1, :, :]))
    result_rev[n - idx - 1, :, :] = np.squeeze(cv2.perspectiveTransform(pts_last[None, :, :], tform_rev))

plt.plot(result[:, 1, 1], 'r-')
plt.plot(result_rev[:, 1, 1], 'b-')
plt.show()


# d1 = np.load('transforms02.npz')
#
# t0 = d0['transforms']
# t1 = d1['transforms']
# f, ax = plt.subplots(3, 3)
# for i in range(3):
#     for j in range(3):
#         x0 = t0[:1000, i, j]
#         x1 = t1[:1000, i, j]
#         ax[i, j].plot(x0, 'b-')
#         ax[i, j].plot(x1, 'r-')
# plt.show()

