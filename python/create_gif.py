import cv2
import imageio

start_frame = 8100
end_frame = 8190
filename = 'data/test_00.mp4'
file2write = 'docs/images/sample_video.gif'
cap = cv2.VideoCapture(filename)
# get video parameters
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
framePS = cap.get(cv2.CAP_PROP_FPS)

print(frameCount)

cap.set(1, start_frame)
cv2.namedWindow('test', cv2.WINDOW_NORMAL)
frames = []
while True:
    if start_frame < end_frame and start_frame < frameCount:
        start_frame += 1
    else:
        break
    ret, frame = cap.read()
    if not ret:
        break
    resize = cv2.resize(frame, dsize=(frameWidth // 4, frameHeight // 4), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('test', resize)
    frames.append(cv2.cvtColor(resize, cv2.COLOR_BGR2RGB))
    cv2.waitKey(3)


cap.release()
cv2.destroyAllWindows()
imageio.mimsave(file2write, frames, 'GIF', fps=framePS, subrectangles=True)
