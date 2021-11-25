##
import cv2
import numpy as np
import random


def find_points(image1, image2):
    g1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    list = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.6 * n.distance:
            list.append([float(m.distance / n.distance), m, n])
            good.append(m)

    # getting location of matches
    good_points = []
    for i in range(len(good)):
        good_points.append([[int(kp1[good[i].queryIdx].pt[0]), int(kp1[good[i].queryIdx].pt[1])],
                            [int(kp2[good[i].trainIdx].pt[0]), int(kp2[good[i].trainIdx].pt[1])]])

    src_points = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_points = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    return [good_points, src_points, dst_points]


def compute_epipole(matrix):
    U, S, V = np.linalg.svd(matrix)

    res_e = V[-1]
    return res_e / res_e[2]


def draw_matches(image1, image2, good_points, mask_F):

    for i in range(len(good_points)):
        p = good_points[i]
        p1 = p[0]
        p2 = p[1]

        if mask_F[i] == 1:
            c = (0, 255, 0)
        else:
            c = (0, 0, 255)

        cv2.circle(image1, (p1[0], p1[1]), 4, c, 4)
        cv2.circle(image2, (p2[0], p2[1]), 4, c, 4)

    res = np.zeros((image1.shape[0], image1.shape[1] + image2.shape[1], 3))
    res[:, 0:image1.shape[1], :] = image1
    res[0:image2.shape[0], image1.shape[1]:image1.shape[1] + image2.shape[1], :] = image2

    return res


def draw_epipole_point(point_e, image):
    e_x = int(point_e[1])
    e_y = int(point_e[0])

    if e_x < 0:
        l_x = -e_x + image.shape[0] + 100 + 100
        new_x = -e_x + 200

    elif e_x > image.shape[0]:
        l_x = e_x + 300
        new_x = 0

    else:
        l_x = image.shape[0] + 100
        new_x = 0

    if e_y < 0:
        l_y = -e_y + image.shape[1] + 100 +100
        new_y = -e_y + 200

    elif e_y > image.shape[1]:
        l_y = e_y + 300
        new_y = 0

    else:
        l_y = image.shape[1] + 100
        new_y = 0

    frame = np.ones((l_x, l_y, 3))
    frame = frame * 255
    frame[new_x:new_x + image.shape[0], new_y:new_y + image.shape[1], :] = image
    cv2.circle(frame, (new_y + e_y,new_x + e_x), 50, (0, 0, 255), 50)

    return frame


def epipole_line(im, point, matrix, color, flag):
    line = matrix.dot(point)

    a = -line[0]
    b = -line[1]
    c = -line[2]

    if flag:
        x = 0
        y = -int((a * x + c) / b)
    else:
        x = im.shape[1]
        y = -int((a * x + c) / b)

    if flag:
        x2 = im.shape[1]
        y2 = -int((a * x2 + c) / b)
    else:
        x2 = 0
        y2 = -int((a * x2 + c) / b)

    cv2.line(im, (x, y), (x2, y2), color, 3)

    return im


def all_lines(match_points, mask_F, image1, image2, fundamental, transpose):
    cnt = 0

    for i in range(0, len(match_points), 12):
        p = match_points[i]

        p1 = p[0]
        pr_point1 = np.ones((3, 1))
        pr_point1[0] = p1[0]
        pr_point1[1] = p1[1]

        p2 = p[1]
        pr_point2 = np.ones((3, 1))
        pr_point2[0] = p2[0]
        pr_point2[1] = p2[1]

        if mask_F[i] == 1:
            r = random.randint(0, 255)
            b = random.randint(0, 255)
            g = random.randint(0, 255)
            color = (r, g, b)

            cnt += 1

            image2 = epipole_line(image2, pr_point1, fundamental, color, False)
            cv2.circle(image1, (p1[0], p1[1]), 10, color, 20)

            image1 = epipole_line(image1, pr_point2, transpose, color, True)
            cv2.circle(image2, (p2[0], p2[1]), 10, color, 20)

        if cnt == 10:
            break

    return image1, image2


##
# finding fundamental matrix
im1 = cv2.imread('01.jpg')
im2 = cv2.imread('02.jpg')

matches, pts1, pts2 = find_points(im1, im2)
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 2, 0.99)
print(F)


##

points_drawn = draw_matches(im1, im2, matches, mask)
cv2.imwrite('res05.jpg', points_drawn)


##
e = compute_epipole(F)
e_prim = compute_epipole(F.transpose())
print("e is:",e)
print("e prim is:",e_prim)

##
src1 = cv2.imread('01.jpg')
src2 = cv2.imread('02.jpg')

f1 = draw_epipole_point(e, src1)
cv2.imwrite('res06.jpg', f1)

f2 = draw_epipole_point(e_prim, src2)
cv2.imwrite('res07.jpg', f2)

##

# ff1 = cv2.cvtColor(f1.astype(np.float32), cv2.COLOR_BGR2RGB)
# ff2 = cv2.cvtColor(f2.astype(np.float32), cv2.COLOR_BGR2RGB)
#
# f, axs = plt.subplots(2)
# axs[0].imshow(ff1.astype(np.uint8))
# axs[1].imshow(ff2.astype(np.uint8))
# plt.show()
# plt.savefig('epipole_points.jpg')

##
res1, res2 = all_lines(matches, mask, src1, src2, F, F.transpose())

res = np.zeros((res1.shape[0], res2.shape[1] + res2.shape[1], 3))
res[:, 0:res1.shape[1], :] = res1
res[0:res2.shape[0], res1.shape[1]:res1.shape[1] + res2.shape[1], :] = res2

cv2.imwrite('res08.jpg', res)




##

