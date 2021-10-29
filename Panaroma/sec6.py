##
import cv2
import numpy as np
import time


def foreground_mask(back, frame):
    b1 = back[:, :, 0]
    b2 = back[:, :, 1]
    b3 = back[:, :, 2]

    f1 = frame[:, :, 0]
    f2 = frame[:, :, 1]
    f3 = frame[:, :, 2]

    thresh = 60
    mask1 = np.where((np.add(f1, ((-1)*b1)) > thresh), 1, 0)

    mask2 = np.where((np.add(f2, ((-1)*b2)) > thresh), 1, 0)
    mask3 = np.where((np.add(f3, ((-1)*b3)) > thresh), 1, 0)

    final = np.where((mask1 == 1) & (mask2 == 1) & (mask3 == 1), 255, 0)

    # eroding noise
    kernel = np.ones((3, 5), np.uint8)
    m = cv2.erode(np.float32(final), kernel, iterations=1)

    kernel = np.ones((15, 15), np.uint8)
    mask_d = cv2.dilate(m, kernel, iterations=1)

    d = np.where((mask_d != 0) & (final == 255), 255, 0)

    return d


def color_red(back, frame):
    f = foreground_mask(back, frame)

    res = frame.copy()

    res[:, :, 0] = np.where((f == 255),0, frame[:, :, 0])
    res[:, :, 1] = np.where((f == 255),0, frame[:, :, 1])

    res[:, :, 2] = np.where((f == 255),255, frame[:, :, 2])

    return res


##
cap = cv2.VideoCapture('video.mp4')
cap2 = cv2.VideoCapture('res07-background-video.mp4')

size = (1920, 1080)
out = cv2.VideoWriter('res08-foreground-video.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

cnt = 1
cp = time.time()

while True:
    ret1, frame1 = cap.read()
    ret2, frame2 = cap2.read()

    fore_ground = color_red(frame2, frame1)

    out.write(fore_ground)

    if cnt == 899:
        break
    cnt += 1

out.release()
cap.release()
cap2.release()

cv2.destroyAllWindows()

print(time.time()-cp)
