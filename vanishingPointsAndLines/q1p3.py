##
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


z_theta = 0.04015901932359344
x_theta = 0.1


rotation1 = np.zeros((3,3))
rotation1[0][0] = 1
rotation1[1][2] = -math.sin(x_theta)
rotation1[2][1] = math.sin(x_theta)
rotation1[1][1] = math.cos(x_theta)
rotation1[2][2] = math.cos(x_theta)


rotation2 = np.zeros((3, 3))
rotation2[2][2] = 1
rotation2[0][1] = -math.sin(z_theta)
rotation2[1][0] = math.sin(z_theta)
rotation2[0][0] = math.cos(z_theta)
rotation2[1][1] = math.cos(z_theta)


R = rotation1.dot(rotation2)

K = [[13307,     0,  2158],
    [0, 13307,  1509],
     [0,    0,     1]]
K = np.array(K)

H = K.dot(R).dot(np.linalg.inv(K))
H /= H[2][2]

print(H)
T = np.zeros((3, 3))
T[0][0] = 1
T[1][1] = 1
T[2][2] = 1
T[0][2] = 1*800
T[1][2] = 1*1800

im = cv2.imread('vns.jpg')
w = cv2.warpPerspective(im, T.dot(H), (int((3/2)*im.shape[1]), int((3/2)*im.shape[0])))
cv2.imwrite('res04.jpg',w)


##

