import os
import cv2

video_file = '../data/test_00.mp4'
output_file_pattern = '../data/gt/frame_%06d.jpg'

cap = cv2.VideoCapture(video_file)

frame_skip = 150

if not os.path.exists('../data/gt'):
    os.makedirs('../data/gt')

# get video parameters
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
framePS = cap.get(cv2.CAP_PROP_FPS)


for idx in range(0, frameCount, frame_skip):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        break
    filename = output_file_pattern % idx
    cv2.imwrite(filename, frame)
    print('written %s' % filename)

cap.release()

