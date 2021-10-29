import numpy as np
import cv2
import random


def get_random_points(matches):
    n = len(matches)
    rand1 = random.randint(0, n - 1)
    p1 = matches[rand1][0]
    q1 = matches[rand1][1]

    rand2 = random.randint(0, n - 1)
    p2 = matches[rand2][0]
    q2 = matches[rand2][1]

    rand3 = random.randint(0, n - 1)
    p3 = matches[rand3][0]
    q3 = matches[rand3][1]

    rand4 = random.randint(0, n - 1)
    p4 = matches[rand4][0]
    q4 = matches[rand4][1]

    return [[p1, p2, p3, p4], [q1, q2, q3, q4]]


def get_vector(point):
    v=[[point[0]],
       [point[1]],
       [1]]
    v=np.array(v)
    return v


def small_A(p1, p2):
    v2 = get_vector(p2)

    A = [[0, 0, 0, -v2[0][0], -v2[1][0], -v2[2][0], p1[1]*v2[0][0], p1[1]*v2[1][0], p1[1]*v2[2][0]],
         [v2[0][0], v2[1][0], v2[2][0], 0, 0, 0, -p1[0]*v2[0][0], -p1[0]*v2[1][0], -p1[0]*v2[2][0]]]

    A = np.array(A)

    return A


def matrix_A(points1, points2):
    p1 = points1[0]
    p2 = points1[1]
    p3 = points1[2]
    p4 = points1[3]

    q1 = points2[0]
    q2 = points2[1]
    q3 = points2[2]
    q4 = points2[3]

    A = [small_A(p1, q1)[0,:],
         small_A(p1, q1)[1,:],
         small_A(p2, q2)[0, :],
         small_A(p2, q2)[1, :],
         small_A(p3, q3)[0, :],
         small_A(p3, q3)[1, :],
         small_A(p4, q4)[0, :],
         small_A(p4, q4)[1, :]]

    A = np.array(A)

    return A


def find_homography(points1,points2):
    A = matrix_A(points1,points2)
    u, s, vh = np.linalg.svd(A, full_matrices=True)

    end = (vh.T).shape[1]
    h = vh.T[:, end-1]

    c = h[8]
    H = [[h[0]/c, h[1]/c, h[2]/c],
         [h[3]/c, h[4]/c, h[5]/c],
         [h[6]/c, h[7]/c, 1]]

    H = np.array(H)

    return H


def distance(p, q):
    a = (p[:].astype(int) - q[:].astype(int)) ** 2
    ssd = np.sum(a)
    return np.sqrt(ssd)


def check_H(all_matches, H, tresh):
    Ps = all_matches[:, 0]
    Qs = all_matches[:, 1]

    cnt = 0

    for i in range(len(Ps)):
        p = Ps[i]
        p = get_vector(p)
        q = Qs[i]
        q = get_vector(q)

        d = distance(p, np.matmul(H, q))
        if (d < tresh):
            cnt+=1

    return cnt


def one_h(all_matches,tresh):
    rand_points = get_random_points(all_matches)

    H = find_homography(rand_points[0], rand_points[1])

    number_h = check_H(all_matches, H, tresh)

    return [number_h, H]


def updtae_N(iter, prev_N, number_h, length, prev_w):
    p = 0.99
    s = 4

    if(iter==0):
        w = 0
    else:
        if(prev_w<(number_h/length)):
            w = number_h/length
        else:
            w = prev_w

    if w==0:
        N=prev_N
    else:
        N=(np.log(1-p))/(np.log(1-w**s))

    return [N, w]


def h_ransac(all_matches, tresh, max_iter):
    length = len(all_matches)

    H_final = []
    best_num = 0

    N,w = updtae_N(0, max_iter, 0, length,0)

    for iter in range(0,max_iter+1):
        numH, H=one_h(all_matches,tresh)

        if iter==0:
            H_final=H

        if numH>best_num:
            H_final=H
            best_num=numH

        N,w=updtae_N(iter,N, numH,length,w)

        if iter>=N:
            return [H_final,N,w]
        if iter >= max_iter:
            return [H_final,N,w]

    return "ERROR"


im1 = cv2.imread('im03.jpg')
im2 = cv2.imread('im04.jpg')


# finding points with sift

gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

orb = cv2.SIFT_create()
kp1, des1 = orb.detectAndCompute(gray1,None)
kp2, des2 = orb.detectAndCompute(gray2,None)

# matching points
bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

good = []
for i, (m, n) in enumerate(matches):
    if m.distance < 0.65*n.distance:
        good.append(m)

good_points=[]
for i in range(len(good)):
    good_points.append([[int(kp1[good[i].queryIdx].pt[0]), int(kp1[good[i].queryIdx].pt[1])], [int(kp2[good[i].trainIdx].pt[0]), int(kp2[good[i].trainIdx].pt[1])]])

np_good = np.array(good_points)


# applying ransac to find homography
H,N,w=h_ransac(np_good, 5, 200000)


print("homography:",H)
print("number of samples:",N)
print("last w:",w)

# applying H and warping
warped = cv2.warpPerspective(im2, H, (im1.shape[1], im1.shape[0]))
cv2.imwrite('res20.jpg',warped)

