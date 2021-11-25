##
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


def line(p1, p2):
    p_l1 = [p1[0], p1[1], 1]
    p_l2 = [p2[0], p2[1], 1]

    return np.cross(p_l1, p_l2)


def all_lines(list_points):
    lines = []
    for i in range(0, len(list_points)-1, 2):
        l = line(list_points[i], list_points[i+1])
        lines.append(l)

    return lines


def intersection(l1, l2):
    p = np.cross(l1, l2)
    p/=p[2]
    return p.astype(int)


def all_intersections(lines):
    cross = []

    for i in range(0, len(lines)):
        l1 = lines[i]
        for j in range(i+1, len(lines)):
            l2 = lines[j]
            p = intersection(l1, l2)
            print(i,j,p)
            cross.append(p)

    return cross


def inliers(numbers, thresh, list_lines):
    inlier_lines = []

    for i in range(0, len(list_lines)):
        l1 = list_lines[i]
        cnt = 0
        for j in range(0, len(list_lines)):
            if i != j:
                l2 = list_lines[j]
                p = intersection(l1, l2)
                for t in range(0, len(list_lines)):
                    print(list_lines[t].dot(p))

                    if (t != i) & (list_lines[t].dot(p) < thresh) & (list_lines[t].dot(p) > -thresh):
                        cnt+=1
        if cnt > numbers:
            inlier_lines.append(l1)

    return inlier_lines


def make_matrix(lines):
    A = np.ones((len(lines), 3))

    for i in range(0, len(lines)):
        A[i][0] = lines[i][0]
        A[i][1] = lines[i][1]
        A[i][2] = lines[i][2]

    return A


def get_vanishing_point(matrix):
    u, s, v = np.linalg.svd(matrix)

    vanishing = v[-1]
    return vanishing / vanishing[2]


def draw_vanishing_point(point_v, im):
    im=cv2.resize(im, (int(im.shape[0]/2), int(im.shape[1]/2)), interpolation=cv2.INTER_AREA)
    e_x = int(point_v[1]/2)
    e_y = int(point_v[0]/2)

    print(e_x, e_y)

    if e_x < 0:
        l_x = -e_x + im.shape[0] + 100 + 10
        new_x = -e_x + 100

    elif e_x > im.shape[0]:
        l_x = e_x + 10
        new_x = 0

    else:
        l_x = im.shape[0] + 10
        new_x = 0

    if e_y < 0:
        l_y = -e_y + im.shape[1] + 100 +10
        new_y = -e_y + 100

    elif e_y > im.shape[1]:
        l_y = e_y + 10
        new_y = 0

    else:
        l_y = im.shape[1] + 10
        new_y = 0

    frame = np.zeros((l_x, l_y, 3))
    frame[new_x:new_x + im.shape[0], new_y:new_y + im.shape[1], :] = im
    cv2.circle(frame, (new_x + e_x, new_y + e_y), 10, (0, 0, 255), 50)

    return frame


def draw_all_van_points(src1, Vx, Vy, Vz):
    src = cv2.resize(src1, (int(src1.shape[1] / 4), int(src1.shape[0] / 4)), interpolation=cv2.INTER_AREA)
    vx1 = int(Vx[0]/4)
    vx2 = int(Vx[1]/4)

    vy1 = int(Vy[0]/4)
    vy2 = int(Vy[1]/4)

    vz1 = int(Vz[0] / 4)
    vz2 = int(Vz[1] / 4)

    lx = -vy1 + src.shape[1] + vx1 + 200
    ly = -vz2 + src.shape[0]+ vy2 + 200

    new_x = (-vy1 + 120)
    new_y = (-vz2 + 120)

    frame = np.ones((ly, lx, 3))
    frame = frame * 255
    frame[new_y:new_y + src.shape[0], new_x:new_x + src.shape[1], :] = src

    cv2.circle(frame, (new_x + vx1, new_y + vx2), 60, (0, 0, 255), 100)
    cv2.circle(frame, (new_x + vy1, new_y + vy2), 60, (0, 255, ), 100)
    cv2.circle(frame, (new_x + vz1, new_y + vz2), 60, (255, 0, 0), 100)
    cv2.line(frame, (new_x + vx1, new_y + vx2), (new_x + vy1, new_y + vy2), (0, 0, 255), 8, lineType=8)


    return frame


def solve_k(Vx, Vy, Vz):
    Vx /= Vx[2]
    a1 = Vx[0]
    b1 = Vx[1]

    Vy /= Vy[2]
    a2 = Vy[0]
    b2 = Vy[1]

    Vz /= Vz[2]
    a3 = Vz[0]
    b3 = Vz[1]

    A = a2*(a1-a3) + b2*(b1-b3)
    B = a1*(a2-a3) + b1*(b2-b3)
    C = (((a2-a3)*(b3-b1))/(a1-a3)) + (b2-b3)

    py = (B - ((a2-a3)*A)/(a1-a3)) / C
    px = (A + py*(b3-b1))/(a1-a3)

    f = math.sqrt(-(px**2)-(py**2)+ (a1+a2)*px + (b1+b2)*py - (a1*a2+b1*b2))

    k = np.zeros((3,3))
    k[2][2] = 1
    k[0][0] = f
    k[1][1] = f
    k[0][2] = px
    k[1][2] = py

    return k


def draw_line(Vx, Vy, src):
    l = np.cross(Vx, Vy)

    a1 = -l[0]
    b1 = -l[1]
    c1 = -l[2]

    lx = src.shape[0]+300
    ly = src.shape[1]+300

    x = 0
    y = -(a1 * x + c1) / b1

    xx = ly
    yy = -(a1 * xx + c1) / b1

    res_frame = np.ones((lx, ly, 3))
    res_frame = res_frame*255
    res_frame[0:src.shape[0], 150:150+src.shape[1], :] = src

    cv2.line(res_frame, (int(x), int(y)), (int(xx), int(yy)), (0, 0, 255), 8, lineType=8)

    return res_frame


##
# z-axis
image = cv2.imread('vns.jpg')
img = image.copy()
# blurredSrc = cv2.medianBlur(img, 27)
graySrc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(graySrc, 400, 450, apertureSize=3)
kernel = np.ones((9, 1), np.uint8)
erosion = cv2.erode(canny, kernel, iterations=1)

lines = cv2.HoughLines(erosion, 1, np.pi / 180, 210, None, 0, 0)

z_lines = []
for i in range(0, len(lines)):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 10000 * (-b)), int(y0 + 10000 * (a)))
    pt2 = (int(x0 - 10000 * (-b)), int(y0 - 10000 * (a)))
    cv2.line(image, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    l = line(pt1, pt2)
    z_lines.append(l)

# y_axis
image = cv2.imread('vns.jpg')
img = image.copy()
graySrc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(graySrc, 100, 300, apertureSize=3)
kernel = np.ones((1, 6), np.uint8)
erosion = cv2.erode(canny, kernel, iterations=1)
dilation = cv2.dilate(erosion, kernel)
lines = cv2.HoughLines(dilation, 1, np.pi / 180, 210, None, 0, 0)

y_lines = []
for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 10000 * (-b)), int(y0 + 10000 * (a)))
        pt2 = (int(x0 - 10000 * (-b)), int(y0 - 10000 * (a)))
        cv2.line(image, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

        l = line(pt1, pt2)
        y_lines.append(l)

# x-axis
image = cv2.imread('vns.jpg')
img = image.copy()
graySrc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(graySrc, 448, 449, apertureSize=3)


kernel = np.ones((1, 3), np.uint8)
erosion = cv2.erode(canny, kernel, iterations=1)
dilation = cv2.dilate(erosion, kernel)
lines = cv2.HoughLines(dilation, 1, np.pi / 180, 180, None)

x_lines = []
for i in range(0, len(lines)):
    rho = lines[i][0][0]
    theta = lines[i][0][1]

    if (abs(theta-(math.pi)/2) > 0.15):
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 10000 * (-b)), int(y0 + 10000 * (a)))
        pt2 = (int(x0 - 10000 * (-b)), int(y0 - 10000 * (a)))
        cv2.line(image, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

        l = line(pt1, pt2)
        x_lines.append(l)


##
m_x = make_matrix(x_lines)
m_y = make_matrix(y_lines)
m_z = make_matrix(z_lines)
v_x = get_vanishing_point(m_x)
v_y = get_vanishing_point(m_y)
v_z = get_vanishing_point(m_z)


##
image = cv2.imread("vns.jpg")


K = solve_k(v_x, v_y, v_z)
print(K.astype(int))

##
px = K[0][2]
py = K[1][2]
print("principal_point:",(px,py))
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

res_center = cv2.circle(rgb, (int(px), int(py)), 10, (255, 0, 0), 10)
fig, axs = plt.subplots(1)
axs.imshow(res_center)
axs.set_title('f='+str(int(K[0][0])))
plt.show()
plt.savefig('res03.jpg')

##
tan = float(v_y[1]-v_x[1])/float(v_y[0]-v_x[0])
angle_vertical = math.atan(tan)
print("first angle:", angle_vertical)


##
back_project = np.linalg.inv(K).dot(v_z)

main_z = np.zeros((3, 1))
main_z[0][0] = 0
main_z[1][0] = 0
main_z[2][0] = 1


unit1 = back_project / np.linalg.norm(back_project)
unit2 = main_z / np.linalg.norm(main_z)
dot_product = np.dot(unit1, unit2)


angle_horizontal = math.acos(dot_product)
z_theta = math.pi/2 - angle_horizontal

print("second angle:", z_theta)


##

