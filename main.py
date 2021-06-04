import cv2
import numpy as np
import random


def get_grad(dev_arr):
    xx = dev_arr[0]
    yy = dev_arr[1]

    grad = np.sqrt((xx ** 2) + (yy ** 2)).astype(np.float32)

    grad = grad * (255 / np.max(grad))

    return grad


def get_dev(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    i_x = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)
    i_x = np.where(i_x < 30, 0, i_x)

    i_y = cv2.Sobel(gray, cv2.CV_8U, 0, 1, 3)
    i_y = np.where(i_y < 30, 0, i_y)

    i_xx = i_x * i_x
    i_yy = i_y * i_y
    i_xy = i_x * i_y

    return [i_xx, i_yy, i_xy]


def guass(inp):
    res=cv2.GaussianBlur(inp,(3,3),5,5, cv2.BORDER_CONSTANT)
    return res


def component(image, num, labels):
    result = np.zeros(image.shape)

    for l in range(0, num):
        sub = np.where(labels==l+1, image, 0)
        maximum=np.amax(sub)
        result = np.where(((image==maximum) & (labels==l+1)), 1, result)

    return result


def get_FVector(image, point):
    n = 23

    x = point[0]
    y = point[1]

    v = image[x-n:x+n+1,y-n:y+n+1,:]

    return v


def find_match(v1, dist_arr):
    ds = []
    m = len(dist_arr)

    d1 = ssd(v1, dist_arr[0])
    index1 = 0

    d2 = ssd(v1, dist_arr[0])

    for cnt in range(0, m):
        value = ssd(v1, dist_arr[cnt])
        ds.append(value)

        if (value < d1):
            d2 = d1
            d1 = value
            index1 = cnt

        elif (value < d2):
            d2 = value


    if (d2 == 0):
        if (d1 == 0):
            return index1
        else:
            return -1

    tresh = 0.78

    if (d1 / d2) < tresh:
        return index1

    else:
        return -1


def ssd(v1, v2):
    A = (v1[:, :, :].astype(int)-v2[:, :, :].astype(int))**2
    ssd = np.sum(A)

    return ssd


def all_vectors(image, points):
    vectors = []
    m = len(points)

    for cnt in range(0, m):
        v = get_FVector(image, points[cnt])
        vectors.append(v)

    return vectors


def array_match(vecs1, vecs2):
    matches = []

    m = len(vecs1)

    for cnt in range(0, m):
        p_index = find_match(vecs1[cnt], vecs2)
        matches.append([cnt, p_index])

    return matches


def get_good_matches(match1,match2):
    match_result=[]
    length=len(match1)

    for cnt in range(0,length):
        dst=match1[cnt][1]
        if (dst!=-1) & (match2[dst][1]==cnt):
            match_result.append([cnt,dst])

    return match_result


im1 = cv2.imread('im01.jpg')
im2 = cv2.imread('im02.jpg')

arr1 = get_dev(im1)
g1 = get_grad(arr1)
cv2.imwrite('res01_grad.jpg',g1)

arr2 = get_dev(im2)
g2 = get_grad(arr2)
cv2.imwrite('res02_grad.jpg',g2)

s1_xx = guass(arr1[0])
s1_yy = guass(arr1[1])
s1_xy = guass(arr1[2])


s2_xx = guass(arr2[0])
s2_yy = guass(arr2[1])
s2_xy = guass(arr2[2])


det1 = s1_xx*s1_yy-s1_xy
trace1 = s1_xx+s1_yy

det2 = s2_xx*s2_yy-s2_xy
trace2 = s2_xx+s2_yy


k = 0.47
R1 = det1-k*trace1
R2 = det2-k*trace2

cv2.imwrite('res03_source.jpg',R1)
cv2.imwrite('res04_source.jpg',R2)


# tresh
R1_prim = np.where(R1>140, R1, 0)
R2_prim = np.where(R2>140, R2, 0)

cv2.imwrite('res05_thresh.jpg', R1_prim)
cv2.imwrite('res06_thresh.jpg', R2_prim)

binary1 = np.where(R1_prim==0, 0, 255)
binary2 = np.where(R2_prim==0, 0, 255)

binary1 = cv2.GaussianBlur(np.uint8(binary1), (9, 9), 7, 7, cv2.BORDER_CONSTANT)
binary1 = np.where(binary1==0, 0, 1)

binary2 = cv2.GaussianBlur(np.uint8(binary2), (9, 9), 7, 7, cv2.BORDER_CONSTANT)
binary2 = np.where(binary2==0,0,1)

num1, labels1 = cv2.connectedComponents(np.uint8(binary1))
num2, labels2 = cv2.connectedComponents(np.uint8(binary2))


non_max1=component(R1_prim,num1,labels1)
non_max2=component(R2_prim,num2,labels2)



minimum = 21
max_xx = im2.shape[0]-minimum
max_yy = im2.shape[1]-minimum

dots1 = im1
points1 = []

for i in range(0, im1.shape[0]):
    for j in range(0, im1.shape[1]):
        if (non_max1[i][j] != 0) & ((i<max_xx) & (j<max_yy) & (i > minimum) & (j > minimum)):
            cv2.circle(dots1, (j, i), 3, (0, 0, 255), 3)
            points1.append([i, j])
print("circled1")
cv2.imwrite('res07_harris.jpg', dots1)

dots2 = im2
points2 = []

for i in range(0, im2.shape[0]):
    for j in range(0, im2.shape[1]):
        if (i<max_xx) & (j<max_yy) & (i > minimum) & (j > minimum) & (non_max2[i][j] != 0):
            cv2.circle(dots2, (j, i), 3, (0, 0, 255), 3)
            points2.append([i, j])
print("circled2")
cv2.imwrite('res08_harris.jpg', dots2)


image1 = cv2.imread('im01.jpg')
image2 = cv2.imread('im02.jpg')
result_final = cv2.hconcat([image1, image2])


vectors1 = all_vectors(image1, points1)
vectors2 = all_vectors(image2, points2)

# finding matches for points of first image
match_im1 = array_match(vectors1,vectors2)
print("done")

# finding matches for points of second image
match_im2 = array_match(vectors2,vectors1)
print("done")

# cross checking
goods = get_good_matches(match_im1,match_im2)


n=len(goods)
w = im1.shape[0]
h = im1.shape[1]
for x in range(0,n):
    match = goods[x]
    index1 = match[0]
    index2 = match[1]

    r = random.randint(0,255)
    b = random.randint(0,255)
    g = random.randint(0,255)
    color = (r, g, b)

    if (index2!=-1):
        cv2.line(result_final, (points1[index1][1],points1[index1][0]), (points2[index2][1]+h,points2[index2][0]), color, 2)
        cv2.circle(image1, (points1[index1][1],points1[index1][0]), 3, (0, 0, 255), 3)
        cv2.circle(image2, (points2[index2][1],points2[index2][0]), 3, (0, 0, 255), 3)


cv2.imwrite('res09.jpg', image1)
cv2.imwrite('res10.jpg', image2)

cv2.imwrite('res11.jpg', result_final)


