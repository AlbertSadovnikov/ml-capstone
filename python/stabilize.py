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
feature_params = dict(maxCorners=256,
                      qualityLevel=0.25,
                      minDistance=32,
                      blockSize=32)

# Parameters for Lucas Kanade optical flow
lk_params = dict(winSize=(32, 32),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Find homography parameters
hg_params = dict(method=cv2.RANSAC,
                 ransacReprojThreshold=0.5)

# Create some random colors
color = np.random.randint(0, 255, (feature_params['maxCorners'], 3))

cv2.namedWindow('Stabilized', cv2.WINDOW_NORMAL)

# Take first frame and find corners in it
ret, base_frame = cap.read()
# cv2.imshow('Stabilized', base_frame)
base_gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(base_gray, mask=None, **feature_params)

currentTransform = np.eye(3)

counter = 0
while True:
    # get good tracking points
    counter += 1
    print('Frame %d' % counter)

    # print('\tDetected %d good key points to track' % len(p0))
    # read next frame
    ret, frame = cap.read()
    if not ret:
        break
    # apply current transformation
    correctedFrame = cv2.warpPerspective(frame,
                                         currentTransform,
                                         (1920, 1080),
                                         flags=cv2.INTER_LANCZOS4,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 255))
    # convert to gray
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(base_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_p1 = p1[st == 1]
    good_p0 = p0[st == 1]
    res, inliers = cv2.findHomography(good_p1, good_p0, **hg_params)
    inlierNum = np.sum(inliers)
    print('\t%d inliers out of %d(%d)' % (inlierNum, len(good_p0), len(p0)))
    currentTransform = np.dot(currentTransform, res)
    cv2.imshow('Stabilized', correctedFrame)
    cv2.waitKey(1)
    # copy
    base_gray = frame_gray.copy()
    p0 = p1.copy()

cv2.destroyAllWindows()
cap.release()
