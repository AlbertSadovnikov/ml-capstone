import cv2

# parameters for Shi Tomasi corner detection
feature_params = dict(maxCorners=4096,
                      qualityLevel=0.15,
                      minDistance=32,
                      blockSize=32)

# Parameters for Lucas Kanade optical flow
lk_params = dict(winSize=(32, 32),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Find homography parameters
hg_params = dict(method=cv2.RANSAC,
                 ransacReprojThreshold=0.5)

sp_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def get_transform(frame0, frame1, mask=None, points=None):
    # find good corners
    p0 = cv2.goodFeaturesToTrack(frame0, mask=None, **feature_params)
    # subpixel accuracy
    p0 = cv2.cornerSubPix(frame0, p0, (9, 9), (-1, -1), sp_criteria)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(frame0, frame1, p0, None, **lk_params)
    # Select good points
    good_p1 = p1[st == 1]
    good_p0 = p0[st == 1]
    res, _ = cv2.findHomography(good_p1, good_p0, **hg_params)
    return res
