##
import cv2
import numpy as np
import time


def load_matrix(file_name):

    matrix = np.loadtxt("homographies/"+file_name+".txt").reshape(3, 3)

    return matrix


def load_frame(num):
    frame = cv2.imread('frames/num'+str(num)+".jpg")
    return frame


def inv_warp(num):
    s1 = np.zeros((3, 3))
    s1[2][2] = 1
    s1[0][0] = 0.5
    s1[1][1] = 0.5

    s2 = np.zeros((3, 3))
    s2[2][2] = 1
    s2[0][0] = 2
    s2[1][1] = 2

    if num!=450:
        H = load_matrix(str(num)+"-450")
        B = (s2.dot(H)).dot(s1)
        inv_H = np.linalg.inv(B)
        A = inv_H.dot(inv_T)

    else:
        A = inv_T

    w = cv2.warpPerspective(backgournd, A, (int((3/2)*size[0]), int(size[1])))

    return w


def make_video():
    w = inv_warp(1)

    size2 = (w.shape[1], w.shape[0])

    out = cv2.VideoWriter('res09-background-video-wider.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30, size2)

    for i in range(1, 480):
        w = inv_warp(i)
        out.write(w)

    out.release()

    return


##
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