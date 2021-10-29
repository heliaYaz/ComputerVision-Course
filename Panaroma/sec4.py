##
import cv2
import numpy as np
import time


def part(x, y):
    cap = cv2.VideoCapture('res05-reference-plane.mp4')

    frames = []
    cnt = 1

    flag1 = False
    flag2 = False

    if y == 1:
        flag1 = True
        limit = 700
    elif y == 2:
        flag2 = True
        limit = 900
    else:
        limit = 600

    while True:
        if (flag1) & (cnt < 250):
            cap.read()
            cnt += 1
        if (flag2) & (cnt < 550):
            cap.read()
            cnt += 1
        else:
            if cnt % 4 != 0:
                cnt += 1
                cap.read()
            else:
                ret, frame = cap.read()
                frames.append(frame[int((x / 3) * frame.shape[0]):int(((x + 1) / 3) * frame.shape[0]), int((y / 3) * frame.shape[1]):int(((y + 1) / 3) * frame.shape[1]), :])
                cnt += 1

        if cnt >= limit:
            break

    frames = np.ma.masked_equal(frames, 0)
    medianFrame = np.ma.median(frames, axis=0).filled(0)

    return medianFrame


##
cp = time.time()
p1 = part(0, 0)
cv2.imwrite('part1.jpg', p1)
print(time.time()-cp)


##
cp = time.time()
p2 = part(0, 1)
cv2.imwrite('part2.jpg', p2)
print(time.time()-cp)


##
cp = time.time()
p3 = part(0, 2)
cv2.imwrite('part3.jpg', p3)
print(time.time()-cp)


##
cp = time.time()
p4 = part(1, 0)
cv2.imwrite('part4.jpg', p4)
print(time.time()-cp)

##
cp = time.time()
p5 = part(1, 1)
cv2.imwrite('part5.jpg', p5)
print(time.time()-cp)

##
cp = time.time()
p6 = part(1, 2)
cv2.imwrite('part6.jpg', p6)
print(time.time()-cp)

##
cp = time.time()
p7 = part(2, 0)
cv2.imwrite('part7.jpg', p7)
print(time.time()-cp)

##
cp = time.time()
p8 = part(2, 1)
cv2.imwrite('part8.jpg', p8)
print(time.time()-cp)

##
cp = time.time()
p9 = part(2, 2)
cv2.imwrite('part9.jpg', p9)
print(time.time()-cp)



##
p1 = cv2.imread('part1.jpg')
p2 = cv2.imread('part2.jpg')
p3 = cv2.imread('part3.jpg')
p4 = cv2.imread('part4.jpg')
p5 = cv2.imread('part5.jpg')
p6 = cv2.imread('part6.jpg')
p7 = cv2.imread('part7.jpg')
p8 = cv2.imread('part8.jpg')
p9 = cv2.imread('part9.jpg')


up = np.hstack((p1, p2, p3))
mid = np.hstack((p4, p5, p6))
down = np.hstack((p7, p8, p9))
res = np.vstack((up, mid, down))

cv2.imwrite('res06-background-panorama.jpg', res)

