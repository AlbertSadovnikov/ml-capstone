import cv2
import numpy as np
from argparse import ArgumentParser


parser = ArgumentParser(description='Drone traffic video stabilization')
parser.add_argument('--video', help='Path to video file', dest='video_filename', required=True)
parser.add_argument('--output', help='Path to output video file', dest='out_filename', required=True)
args = parser.parse_args()

cap = cv2.VideoCapture(args.video_filename)

# get video parameters
frameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
framePS = cap.get(cv2.CAP_PROP_FPS)

print('-' * 50)
print('Processing %s' % args.video_filename)
print(' %d frames %d x %d @ %.2f fps' % (frameCount, frameWidth, frameHeight, framePS))
print('-' * 50)

# parameters for Shi Tomasi corner detection
feature_params = dict(maxCorners=4096,
                      qualityLevel=0.1,
                      minDistance=16,
                      blockSize=16)

# Parameters for Lucas Kanade optical flow
lk_params = dict(winSize=(16, 16),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (feature_params['maxCorners'], 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

print('Got %d points to track' % len(p0))
# Create a mask image for drawing purposes

mask = np.zeros_like(old_frame)
while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)
    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()
