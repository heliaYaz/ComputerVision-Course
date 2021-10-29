import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


X = []
Y = []
Z = []
n = []


def load_matrix(file_name):

    matrix = np.loadtxt("homographies/"+file_name+".txt").reshape(3, 3)

    return matrix


def load_frame(num):
    frame = cv2.imread('frames/num'+str(num)+".jpg")
    return frame


def find_angle(num):
    if num == 450:
        H = np.identity(3)
    else:
        H = load_matrix(str(num)+"-450")

    A = (s2.dot(H)).dot(s1)
    p = A.dot(center)

    p[0] += (shape[0]/2)
    p[1] += (shape[1]/2)

    p = p/p[2]

    norm = math.sqrt(p[0]**2 + p[1]**2 + p[2]**2)

    return [math.acos((p[0])/(norm)), math.acos((p[1])/(norm)), math.acos((p[2])/(norm))]


def all_angles():
    for i in range(1, 901):
        x, y, z = find_angle(i)
        X.append(x)
        Y.append(y)
        Z.append(z)
        n.append(i)
    return


def find_R(x, y, z, num):
    Rx = np.zeros((3, 3))
    Rx[0][0] = 1
    Rx[1][1] = math.cos(x)
    Rx[1][2] = -math.sin(x)
    Rx[2][1] = math.sin(x)
    Rx[2][2] = math.cos(x)

    Ry = np.zeros((3, 3))
    Ry[1][1] = 1
    Ry[0][0] = math.cos(y)
    Ry[0][2] = math.sin(y)
    Ry[2][0] = -math.sin(y)
    Ry[2][2] = math.cos(y)

    Rz = np.zeros((3, 3))
    Rz[2][2] = 1
    Rz[0][0] = math.cos(z)
    Rz[0][1] = -math.sin(z)
    Rz[1][0] = math.sin(z)
    Rz[1][1] = math.cos(z)

    R = Rz.dot(Ry)
    R = R.dot(Rx)

    R /= R[2][2]

    T = [[1,  0, -(shape[0]/2)],
         [0, 1, -(shape[1]/2)],
         [0,  0,   1]]

    T = np.array(T)

    A = (T.dot(R)).dot(np.linalg.inv(T))


    return A


def smooth():
    nn=np.array(n)
    xx = np.linspace(nn.min(), nn.max(), 900)

    itp_x = interp1d(n, X, kind='nearest')
    window_size, poly_order = 101, 3
    new_x = savgol_filter(itp_x(xx), window_size, poly_order)

    itp_y = interp1d(n, Y, kind='nearest')
    window_size, poly_order = 101, 3
    new_y = savgol_filter(itp_y(xx), window_size, poly_order)

    itp_z = interp1d(n, Z, kind='nearest')
    window_size, poly_order = 101, 3
    new_z = savgol_filter(itp_z(xx), window_size, poly_order)

    return new_x, new_y, new_z, xx


def plot():
    fig, axs = plt.subplots(3, 1, figsize=(5, 5))
    axs[0].plot(n, X, 'k', label=r"$f(x) = A + B \tanh\left(\frac{x-x_0}{\sigma}\right)$")
    axs[1].plot(n, Y, 'k', label=r"$f(x) = A + B \tanh\left(\frac{x-x_0}{\sigma}\right)$")
    axs[2].plot(n, Z, 'k', label=r"$f(x) = A + B \tanh\left(\frac{x-x_0}{\sigma}\right)$")

    plt.savefig("before.jpg")

    new_x, new_y, new_z, xx = smooth()

    fig, axs = plt.subplots(3, 1, figsize=(5, 5))
    axs[0].plot(xx, new_x, 'k', label=r"$f(x) = A + B \tanh\left(\frac{x-x_0}{\sigma}\right)$")
    axs[1].plot(xx, new_y, 'k', label=r"$f(x) = A + B \tanh\left(\frac{x-x_0}{\sigma}\right)$")
    axs[2].plot(xx, new_z, 'k', label=r"$f(x) = A + B \tanh\left(\frac{x-x_0}{\sigma}\right)$")

    plt.savefig("after.jpg")
    return


def correct():
    tran_mat = []
    thresh1 = 0.005
    thresh2 = 0.01

    for i in range(0, 900):
        theta = X[i]-X2[i]
        alpha = Y[i]-Y2[i]
        gamma = Z2[i]-Z[i]
        # if abs(gamma) > thresh1:
        #     gamma = gamma/100
        #     # gamma = (gamma/abs(gamma))*thresh1
        # if abs(theta) > thresh2:
        #     # theta = theta/2
        #     theta /=100
        # if abs(alpha) > thresh2:
        #     alpha /=100
        #     # alpha = (alpha / abs(alpha)) * thresh2

        # t = find_R(theta, alpha, gamma, i+1)
        t = find_R(0, 0, gamma, i+1)

        tran_mat.append(t)

    return tran_mat


def warp(num, R):
    f = load_frame(num)

    if num != 450:
        h = load_matrix(str(num)+"-450")
    else:
        h = np.identity(3)
    A = (np.linalg.inv(h).dot(R[num-1])).dot(h)
    w = cv2.warpPerspective(f, A, shape)
    return w


def make_video(t_mat):
    out = cv2.VideoWriter('res10-video-shakeless.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30, shape)

    for i in range(1, 900):
        w = warp(i, t_mat)
        out.write(w)

    out.release()

    return


s1 = np.zeros((3, 3))
s1[2][2] = 1
s1[0][0] = 0.5
s1[1][1] = 0.5
s2 = np.zeros((3, 3))
s2[2][2] = 1
s2[0][0] = 2
s2[1][1] = 2


shape = (1920, 1080)

center = [[shape[0]/2],
          [shape[1]/2],
          [1]]

point = [[shape[0]],
          [shape[1]/2],
          [1]]

all_angles()
X2, Y2, Z2, n2 = smooth()

plot()
translations = correct()
make_video(translations)