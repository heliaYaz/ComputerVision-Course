##
import numpy as np
import cv2
import time


def save_matrix(file_name, matrix):
    a_file = open("homographies/"+file_name+".txt", "w")

    for row in matrix:
        np.savetxt(a_file, row)

    a_file.close()
    return


def load_matrix(file_name):
    matrix = np.loadtxt("homographies/"+file_name+".txt").reshape(3, 3)

    return matrix


def get_homography(image1, image2):
    src_points, dst_points = find_points(image1, image2)
    H, m = cv2.findHomography(dst_points, src_points, cv2.RANSAC, 5, None, maxIters=20000, confidence=0.97)

    return H


def find_points(image1, image2):
    g1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    # surf = cv2.xfeatures2d.SURF_create()

    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)

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


def new_h(frame_num):
    f = load_frame(frame_num)
    f = resize(f, 50)

    if frame_num < 270:
        h = get_homography(f270, f)
        H = H270.dot(h)
        save_matrix(str(frame_num)+"-450", H)
        return

    if frame_num > 630:
        h = get_homography(f630, f)
        H = H630.dot(h)
        save_matrix(str(frame_num)+"-450", H)
        return

    H = get_homography(f450,f)
    save_matrix(str(frame_num) + "-450", H)
    return


def sav_all(start, end):
    for i in range(start, end):
        new_h(i)

    return


def warp(num):
    f = load_frame(num)
    # f = resize(f, 50)

    if num == 450:
        w = cv2.warpPerspective(f, T, size)
        return w

    s1 = np.zeros((3, 3))
    s1[2][2] = 1
    s1[0][0] = 0.5
    s1[1][1] = 0.5
    s2 = np.zeros((3, 3))
    s2[2][2] = 1
    s2[0][0] = 2
    s2[1][1] = 2
    H_mat = load_matrix(str(num)+"-450")
    A=(s2.dot(H_mat)).dot(s1)
    w = cv2.warpPerspective(f, T.dot(A), size)

    # w = resize(w, 200)

    return w


def make_video():
    w1=warp(1)
    size2=(w1.shape[1],w1.shape[0])
    out = cv2.VideoWriter('res05-reference-plane.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30, size2)

    for i in range(1, 901):
        w = warp(i)
        out.write(w)

    out.release()

    return


def resize(img, scale):
    scale_percent = scale
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized


##

f630 = resize(load_frame(630), 50)
f270 = resize(load_frame(270), 50)
f450 = resize(load_frame(450), 50)

T = np.zeros((3, 3))
T[0][0] = 1
T[1][1] = 1
T[2][2] = 1
T[0][2] = 1*1800
T[1][2] = 1*450

size = (5 * f450.shape[1]+960, 3*f450.shape[0]+800)


##
new_h(270)
new_h(630)

f = resize(load_frame(270),50)
H = get_homography(f450, f)
save_matrix(str(270) + "-450", H)

H270 = load_matrix('270-450')
H630 = load_matrix('630-450')


##
sav_all(1, 270)

##
sav_all(271, 450)
print("executed")

##
sav_all(451, 630)
print("executed")

##
sav_all(631, 901)
print("executed")


##
cp = time.time()
make_video()
print(time.time()-cp)


