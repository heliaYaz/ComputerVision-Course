import cv2
import numpy as np


logo=cv2.imread('logo.png')

# distance cam1 & cam2
r = np.sqrt(25**2+40**2)

#rotation matrix
R = np.zeros((3,3))

R[0][0]=1
R[1][1]=25/r
R[1][2]=-40/r
R[2][1]=40/r
R[2][2]=25/r
R[0][0]=1


# c
c=np.zeros((3,1))

c[0][0]=0
c[1][0]=-40*(25/r)
c[2][0]=40*(40/r)

t = -R.dot(c)

# k matrix of cam1
k=np.zeros((3,3))
k[0][0]=500
k[1][1]=500
k[0][2]=128
k[1][2]=128
k[2][2]=1

# k matrix of cam2
k2=np.zeros((3,3))
k2[0][0]=500
k2[1][1]=500
k2[0][2]=600
k2[1][2]=800
k2[2][2]=1


# plane vector
n = np.zeros((3,1))
n[1][0] = -(40/r)
n[2][0] = -(25/r)
n2 = n.transpose()

k_inverse = np.linalg.inv(k)

# homography matrix
H = (k2.dot(R-(t.dot(n2)/25))).dot(k_inverse)

logo = logo.astype(np.float32)
result = []
warp_dst = cv2.warpPerspective(logo, H,(1200, 1200))
cv2.imwrite('newLogo.jpg', warp_dst)
