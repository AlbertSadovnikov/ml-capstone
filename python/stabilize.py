import cv2
import numpy as np
from argparse import ArgumentParser
from traffic.stabilize import get_transform

parser = ArgumentParser(description='Drone traffic video stabilization')
parser.add_argument('--video', help='Path to video file', dest='video_filename', required=True)
parser.add_argument('--output', help='Path to output video file', dest='out_filename', required=False)
args = parser.parse_args()

cap = cv2.VideoCapture(args.video_filename)

# get video parameters
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
framePS = cap.get(cv2.CAP_PROP_FPS)

print('-' * 50)
print('Processing %s' % args.video_filename)
print(' %d frames %d x %d @ %.2f fps' % (frameCount, frameWidth, frameHeight, framePS))
print('-' * 50)

ret, frame = cap.read()
base_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

transforms = np.zeros((frameCount, 3, 3), dtype=np.float)
transforms[0, :, :] = np.eye(3)

for idx in range(1, frameCount):
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    transforms[idx, :, :] = get_transform(base_gray, frame_gray)
    base_gray = frame_gray.copy()
    if idx % 100 == 0:
        print('%d frames processed' % idx)

out_file_name = 'transforms02'
np.savez(out_file_name, transforms=transforms)
cap.release()
