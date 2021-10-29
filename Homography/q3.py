import cv2
import numpy as np

im1=cv2.imread('im03.jpg')
im2=cv2.imread('im04.jpg')


# applying sift
gray1=cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
gray2=cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

orb = cv2.SIFT_create()
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

im1_points = cv2.imread('im03.jpg')
im2_points = cv2.imread('im04.jpg')

img1 = cv2.drawKeypoints(im1_points, kp1, None, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

img2 = cv2.drawKeypoints(im2_points, kp2, None, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

res = np.zeros((im1.shape[0], im1.shape[1]+im2.shape[1], 3))
res[:, 0:im1.shape[1], :] = img1
res[0:im2.shape[0], im1.shape[1]:im1.shape[1]+im2.shape[1], :] = img2

cv2.imwrite('res13_corners.jpg', res)


# matching points
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

good = []
matchesMask = [[0, 0] for i in range(len(matches))]
list2 = []
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        list2.append([float(m.distance/n.distance), m, n])
        good.append(m)
        matchesMask[i] = [1, 0]

# getting location of matches
good_points = []
for i in range(len(good)):
    good_points.append([[int(kp1[good[i].queryIdx].pt[0]), int(kp1[good[i].queryIdx].pt[1])], [int(kp2[good[i].trainIdx].pt[0]), int(kp2[good[i].trainIdx].pt[1])]])


# drawing circles of matched points
for i in range(len(good_points)):
    p=good_points[i]
    p1=p[0]
    p2=p[1]
    cv2.circle(img1, (p1[0], p1[1]), 4, (255, 0, 0), 4)
    cv2.circle(img2, (p2[0], p2[1]), 4, (255, 0, 0), 4)


res2=np.zeros((im1.shape[0], im1.shape[1]+im2.shape[1], 3))
res2[:, 0:im1.shape[1], :] = img1
res2[0:im2.shape[0], im1.shape[1]:im1.shape[1]+im2.shape[1], :]=img2
cv2.imwrite('res14_correspondences.jpg',res2)


draw_params = dict(matchColor=(255, 0, 0),
                   singlePointColor=(0, 255, 0),
                   matchesMask=matchesMask,
                   flags=cv2.DrawMatchesFlags_DEFAULT)

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
cv2.imwrite('res15_matches.jpg', img3)


# sorting good matches to get top 20
sort = sorted(list2, key=lambda x: x[0])
best_match = []
for i in range(len(sort)):
    best_match.append(sort[i][1])

img4 = cv2.drawMatches(im1, kp1, im2, kp2, best_match[:20], None, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('res16.jpg', img4)


# finding homography
src_points = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
dst_points = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
H, mask = cv2.findHomography(dst_points, src_points, cv2.RANSAC, 3, None, maxIters=170000, confidence=0.97)
inlier_mask = mask.ravel().tolist()

print(H)

draw_params_inlier = dict(matchColor=(0, 0, 255), singlePointColor=None, matchesMask=inlier_mask, flags=2)

# drawing all matches in blue
img_res = cv2.drawMatches(im1, kp1, im2, kp2, good, None, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

im11 = img_res[:, 0:im1.shape[1], :]
im22 = img_res[0:im2.shape[0], im1.shape[1]:im1.shape[1]+im2.shape[1], :]

# drawing inliner matches in red
result = cv2.drawMatches(im11, kp1, im22, kp2, good, None, **draw_params_inlier)
cv2.imwrite('res17.jpg', result)

# applying homography and warp
warped = cv2.warpPerspective(im2, H, (im1.shape[1], im1.shape[0]))
cv2.imwrite('res19.jpg', warped)

