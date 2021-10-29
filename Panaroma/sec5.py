##
import cv2
import numpy as np
import time


def load_matrix(file_name):
    matrix = np.loadtxt("homographies/"+file_name+".txt").reshape(3, 3)

    return matrix


def inv_warp(num):
    s1 = np.zeros((3, 3))
    s1[2][2] = 1
    s1[0][0] = 0.5
    s1[1][1] = 0.5

    s2 = np.zeros((3, 3))
    s2[2][2] = 1
    s2[0][0] = 2
    s2[1][1] = 2

    if num != 450:
        H = load_matrix(str(num)+"-450")
        inv_H = np.linalg.inv(H)
        B = (s2.dot(inv_H)).dot(s1)
        A = B.dot(inv_T)

    else:
        A = inv_T

    w = cv2.warpPerspective(backgournd, A, size)

    return w


def make_video():
    out = cv2.VideoWriter('res07-background-video.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

    for i in range(1, 900):
        w = inv_warp(i)
        # w = resize(w, 200)
        out.write(w)

    out.release()

    return


backgournd = cv2.imread('res06-background-panorama.jpg')

# inverse translation
T = np.zeros((3, 3))
T[0][0] = 1
T[1][1] = 1
T[2][2] = 1
T[0][2] = 1*1800
T[1][2] = 1*450
inv_T = np.linalg.inv(T)

# video size
size = (1920, 1080)

##
cp = time.time()
make_video()
print(time.time()-cp)
