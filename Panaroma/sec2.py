##
import cv2
import numpy as np


def load_frame(num):
    frame = cv2.imread('frames/num'+str(num)+".jpg")
    return frame


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
        if m.distance < 0.7 * n.distance:
            good.append(m)

    src_points = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_points = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    return [src_points, dst_points]


def get_mask(image):
    m = np.zeros(image.shape)
    m[:, :, 0] = np.where((image[:, :, 0] < 1) & (image[:, :, 1] < 1) & (image[:, :, 2] < 1), 0, 1)
    m[:, :, 1] = m[:, :, 0]
    m[:, :, 2] = m[:, :, 0]

    return m


def get_edge(num,H):
    if num == 450:
        mat = T
    else:
        mat = T.dot(H)

    h, w, d = f450.shape

    src_points = np.float32([[0, 0], [0, h], [w, h], [w, 0]])
    predicted_dst = cv2.perspectiveTransform(src_points.reshape(-1, 1, 2), mat).reshape(src_points.shape)

    p_min = np.min(predicted_dst, axis=0)

    return p_min


def get_final_point(num,H):
    if num == 450:
        mat = T
    else:
        mat = T.dot(H)

    h, w, d = f450.shape

    point = mat.dot([[w],
                    [0],
                    [1]])

    p_max = [point[0]/point[2], point[1]/point[2]]
    return p_max


def warp(num, h):
    if num == 450:
        return cv2.warpPerspective(f450, T, size)

    f = load_frame(num)

    w = cv2.warpPerspective(f, T.dot(h), size)

    return w


# loading key frames
f450 = load_frame(450)

# size of result
size = (3 * f450.shape[1], 2*f450.shape[0])

# Translation matrix:

T = np.zeros((3, 3))
T[0][0] = 1
T[1][1] = 1
T[2][2] = 1
T[0][2] = 1*1200
T[1][2] = 1*500

f270 = load_frame(270)
f630 = load_frame(630)
f90 = load_frame(90)
f810 = load_frame(810)

H_270 = get_homography(f450, f270)
H_630 = get_homography(f450, f630)
h90 = get_homography(f270, f90)
h810 = get_homography(f630, f810)
H_90 = H_270.dot(h90)
H_810 = H_630.dot(h810)


w_90 = warp(90, H_90)
mask90 = get_mask(w_90)


w_270 = warp(270, H_270)
mask270 = get_mask(w_270)

w_630 = warp(630, H_630)
mask630 = get_mask(w_630)

w_810 = warp(810, H_810)
mask810 = get_mask(w_810)

w_450 = warp(450, T)
mask450 = get_mask(w_450)


mask1 = np.where((mask90 != 0) & (mask270 == 0), 1, 0)
mask2 = np.where((mask270 != 0) & (mask450 == 0), 1, 0)
mask3 = np.where((mask630 != 0) & (mask450 == 0), 1, 0)
mask4 = np.where((mask810 != 0) & (mask630 == 0), 1, 0)



##
result = np.zeros((2*f450.shape[0], 3*f450.shape[1], 3))

x = int(3/2 * f450.shape[0])

# stitching : 90 to 270

# find a point on edge (intersection)
p1 = get_edge(450, T)
x1 = int(p1[1])
y1 = int(p1[0])

# building mask
s = 1900
mask = np.zeros((2 * f450.shape[0], 2*f450.shape[1], 3))
mask[x1:x1+x, y1:y1+int(s/4), :] = np.ones((x, int(s/4), 3))
mask[x1:x1+x, y1:y1+int(s), :] = np.where(mask2[x1:x1+x, y1:y1+int(s), :] == 1, 1, mask[x1:x1+x, y1:y1+int(s), :])

mask1_2 = np.zeros(mask1.shape)
mask1_2[x1:x1+x, y1:y1+int(s), :] = cv2.GaussianBlur(np.float32(mask[x1:x1+x, y1:y1+int(s), :]), (391,201), 198)


# building result using mask
result = np.where(mask2 == 1, w_270, result)
result[x1:x1+x, y1:y1+int(s), :] = np.where(mask450[x1:x1+x, y1:y1+int(s), :] == 0, result[x1:x1+x, y1:y1+int(s), :], mask1_2[x1:x1+x, y1:y1+int(s), :]*w_270[x1:x1+x, y1:y1+int(s), :] + (1-mask1_2[x1:x1+x, y1:y1+int(s), :])*w_450[x1:x1+x, y1:y1+int(s), :])
result = np.where(result == 0, w_450, result)


##
# stitching : previous result  to 450

# find a point on edge (intersection)
p2 = get_edge(270, H_270)
x2 = int(p2[1])-60
y2 = int(p2[0])-80

# building mask
s = 1000
mask = np.zeros((2 * f450.shape[0], 2*f450.shape[1], 3))
mask[x2:x2+x, y2:y2+int(s/4), :] = np.ones((x, int(s/4), 3))
mask[x2:x2+x, y2:y2+int(s), :] = np.where(mask1[x2:x2+x, y2:y2+int(s), :] == 1, 1, mask[x2:x2+x, y2:y2+int(s), :])

# here we dilate mask to get a bigger intersection between 2 imgs
kernel = np.ones((19, 39), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=1)

mask2_2 = np.zeros(mask1.shape)
mask2_2[x2:x2+x, y2:y2+int(s), :] = cv2.GaussianBlur(np.float32(mask[x2:x2+x, y2:y2+int(s), :]), (151, 7), 198)

# getting result
result[x2:x2+x, y2:y2+int(s), :] = np.where(mask2_2[x2:x2+x, y2:y2+int(s), :] == 0, result[x2:x2+x, y2:y2+int(s), :], mask2_2[x2:x2+x, y2:y2+int(s), :]*w_90[x2:x2+x, y2:y2+int(s), :] + (1-mask2_2[x2:x2+x, y2:y2+int(s), :])*result[x2:x2+x, y2:y2+int(s), :])
result = np.where((mask1 == 1) & (mask2_2 == 0), w_90, result)


##
# stitching : previous result  to 630
x = int(3/2 * f450.shape[0])+100

# find a point on edge (intersection)
p3 = get_final_point(450, T)
x3 = int(p3[1])-65
y3 = int(p3[0])-500

# getting mask
s = 800
# mask = np.ones((2 * f450.shape[0], 3*f450.shape[1], 3))
# mask[x3:x3+x, y3:y3+int(s), :] = np.where(mask3[x3:x3+x, y3:y3+int(s), :] == 1, 0, mask[x3:x3+x, y3:y3+int(s), :])
# mask[x3:x3+x, y3+int(s/4):y3+int(s), :] = np.ones((x, int(s)-int(s/4), 3))

mask = np.zeros((2 * f450.shape[0], 3*f450.shape[1], 3))
mask[x3:x3+x, y3+int(s/2):y3+int(s), :] = np.ones((x, int(s)-int(s/2), 3))
mask[x3:x3+x, y3:y3+int(s), :] = np.where(mask3[x3:x3+x, y3:y3+int(s), :] == 1, 1, mask[x3:x3+x, y3:y3+int(s), :])

kernel = np.ones((19, 49), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=1)

mask3_2 = np.zeros(mask1.shape)
mask3_2[x3:x3+x, y3:y3+int(s), :] = cv2.GaussianBlur(np.float32(mask[x3:x3+x, y3:y3+int(s), :]), (201, 27), 198)
m=np.where((mask3_2[x3:x3+x, y3:y3+int(s), :] == 0) | (mask450[x3:x3+x, y3:y3+int(s), :] != 0), 1, 0)
# getting res
result[x3:x3+x, y3:y3+int(s), :] = np.where(m==1, result[x3:x3+x, y3:y3+int(s), :], (1-mask3_2[x3:x3+x, y3:y3+int(s), :])*result[x3:x3+x, y3:y3+int(s), :] + (mask3_2[x3:x3+x, y3:y3+int(s), :])*w_630[x3:x3+x, y3:y3+int(s), :])
result = np.where((mask630 == 1) & (mask3_2 == 0) & (mask450 == 0), w_630, result)




##
# stitching : previous result  to 810
x = int(3/2 * f450.shape[0])+100

# find a point on edge (intersection)
p4 = get_final_point(630, H_630)
x3 = int(p4[1])-65
y3 = int(p4[0])-350


# getting mask
s = 800
mask = np.zeros((2 * f450.shape[0], 3*f450.shape[1], 3))
mask[x3:x3+x, y3+int(s/4):y3+int(s), :] = np.ones((x, int(s)-int(s/4), 3))
mask[x3:x3+x, y3:y3+int(s), :] = np.where(mask4[x3:x3+x, y3:y3+int(s), :] == 1, 1, mask[x3:x3+x, y3:y3+int(s), :])

kernel = np.ones((32, 39), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=1)

mask4_2 = np.zeros(mask1.shape)
mask4_2[x3:x3+x, y3:y3+int(s), :] = cv2.GaussianBlur(np.float32(mask[x3:x3+x, y3:y3+int(s), :]), (201, 177), 198)

# result
result[x3:x3+x, y3:y3+int(s), :] = np.where(mask4_2[x3:x3+x, y3:y3+int(s), :] == 0, result[x3:x3+x, y3:y3+int(s), :], mask4_2[x3:x3+x, y3:y3+int(s), :]*w_810[x3:x3+x, y3:y3+int(s), :] + (1-mask4_2[x3:x3+x, y3:y3+int(s), :])*result[x3:x3+x, y3:y3+int(s), :])
result = np.where((mask810 == 1) & (mask4_2 == 0) & (result == 0), w_810, result)


# save
cv2.imwrite('res04-key-frames-panorama.jpg', result)

