import os
import cv2

video_file = '../drone/test_00.mp4'
output_file_pattern = 'images/%06d.jpg'

cap = cv2.VideoCapture(video_file)

# NB Only images with step 150 are labeled (range(0, number_of_frames, 150))
frame_skip = 150

if not os.path.exists('images'):
    os.makedirs('images')

# get video parameters
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
framePS = cap.get(cv2.CAP_PROP_FPS)


for idx in range(75, frameCount, frame_skip):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        break
    filename = output_file_pattern % idx
    cv2.imwrite(filename, frame)
    print('written %s' % filename)

cap.release()

