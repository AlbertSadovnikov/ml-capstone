import cv2
import os

data_path = '../data'
image_filename = 'top_potsdam_3_10_RGB.png'
pos_filename = 'top_potsdam_3_10_RGB_Annotated_Cars.png'
neg_filename = 'top_potsdam_3_10_RGB_Annotated_Negatives.png'

filename = os.path.join(data_path, image_filename)

img = cv2.imread(filename)

cv2.namedWindow('sample', cv2.WINDOW_NORMAL)
cv2.imshow('sample', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
