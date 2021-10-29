import numpy as np
import cv2


def load_matrix(file_name):
    matrix = np.loadtxt(file_name+".txt").reshape(3, 3)

    return matrix


def get_homography(image1, image2):
    src_points, dst_points = find_points(image1, image2)
    H, m = cv2.findHomography(dst_points, src_points, cv2.RANSAC, 3, None, maxIters=170000, confidence=0.97)

    return H


def find_points(image1, image2):
    g1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    orb = cv2.SIFT_create()
    kp1, des1 = orb.detectAndCompute(g1, None)
    kp2, des2 = orb.detectAndCompute(g2, None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            good.append(m)

    src_points = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_points = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    return [src_points, dst_points]


def load_frame(num):
    frame = cv2.imread('frames/num'+str(num)+".jpg")
    return frame


T = np.zeros((3, 3))
T[0][0] = 1
T[1][1] = 1
T[2][2] = 1
T[0][2] = 1*900
T[1][2] = 1*250


f450 = load_frame(450)
f270 = load_frame(270)


H = get_homography(f450, f270)


x1 = [900, 300]
x2 = [1400, 300]
x3 = [1400, 800]
x4 = [900, 800]


pts = np.float32([x1, x2, x3, x4]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, H)
im1 = cv2.polylines(f450, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
im2 = cv2.polylines(f270, [np.int32(pts)], True, 255, 3, cv2.LINE_AA)

cv2.imwrite('res01-450-rect.jpg', im1)
cv2.imwrite('res02-270-rect.jpg', im2)


f1 = load_frame(450)
f2 = load_frame(270)

size=(int(3/2*f2.shape[1]), int(3/2*f2.shape[0]))

w1 = cv2.warpPerspective(f1, T, size)
w2 = cv2.warpPerspective(f2, T.dot(H), size)


# finding point on edge
v = np.zeros((3, 1))
v[2][0] = 1
x = T.dot(v)
dy = int(x[0][0]/x[2][0])
dx = int(x[1][0]/x[2][0])


# feather mask
s = 900
mask = np.zeros((w1.shape[0], dy+w1.shape[1], 3))
mask[dx:dx+f1.shape[0], dy:dy+int(s/2), :] = np.ones((f1.shape[0], int(s/2), 3))
mask[dx:dx+f1.shape[0], dy:dy+s, :] = cv2.GaussianBlur(mask[dx:dx+f1.shape[0], dy:dy+s, :], (391,221), 78)


res = np.zeros((w2.shape[0], dy+w2.shape[1], 3))
res[:, 0:w2.shape[1], :] = w2
res2 = np.zeros((w2.shape[0], dy+w2.shape[1], 3))
res2[dx:dx+f1.shape[0], dy:dy+f1.shape[1], :] = f1

result = np.zeros((w2.shape[0], dy+w2.shape[1], 3))
result[:, 0:w2.shape[1], :] = w2
result[dx:dx+f1.shape[0], dy:dy+f1.shape[1], :] = f1

result[dx:dx+f1.shape[0], dy:dy+s, :] = mask[dx:dx+f1.shape[0], dy:dy+s, :] * res[dx:dx+f1.shape[0], dy:dy+s, :] + (1-mask[dx:dx+f1.shape[0], dy:dy+s, :])*res2[dx:dx+f1.shape[0], dy:dy+s, :]


cv2.imwrite('res03-270-450-panorama.jpg', result)