##
import cv2
import numpy as np
import math
import time
import os.path

##
# global parameters
list_labels = ["Bedroom", "Coast", "Forest", "Highway", "Industrial", "Inside_City", "Kitchen", "Livingroom",
               "Mountain", "Office", "Open_Country", "Store", "Street", "Suburb", "Tall_Building"]
cnt_labels = len(list_labels)
all_mats = []
test_matrices = []


# bug
def distance_mat(vector, f_mat):
    d = f_mat[:, :].astype(np.float64) - vector.astype(np.float64)
    d = np.abs(d)
    dist = np.sum(d, axis=1)

    s = f_mat.shape[0]

    return dist.reshape(s, 1)


def get_vector(image, size):
    vector = image.reshape(size*size)
    return vector


def imageToVector(path, size):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    res_image = cv2.resize(image, (size, size))
    vector = get_vector(res_image, size)
    return vector


def make_mat(label, validation, size, test):
    path = 'Data/Train/' + label

    dirListing = os.listdir(path)

    vector_list = []

    i = 0
    for filename in dirListing:
        if test:
            v = imageToVector(os.path.join(path, filename), size)
            vector_list.append(v)

        else:
            if i % 5 != validation:
                v = imageToVector(os.path.join(path, filename),size)
                vector_list.append(v)

        i += 1

    mat = np.array(vector_list)

    return mat


def all_matrices(validation, size):
    global all_mats
    all_mats = []
    i = 0
    for label in list_labels:
        mat = make_mat(label, validation, size, False)
        all_mats.append(mat)
        i += 1

    return all_mats


def validate(validation, size, k):
    number = 0
    good_cnt = 0

    for label in list_labels:
        good, n = validate_directory(label, validation, size, k)
        good_cnt += good
        number += n

    return (good_cnt/number) * 100


def validate_directory(label, validation, size, k):
    path = 'Data/Train/' + label
    dirListing = os.listdir(path)

    good = 0
    i = 0
    n = 0
    for filename in dirListing:
        if i % 5 == validation:
            n += 1
            dst = check_query(os.path.join(path, filename), size, k, False)

            if dst == label:
                good += 1

        i += 1

    return good, n


def check_query(query_path, size, k, test):
    query_vector = imageToVector(query_path, size)
    neighbors = get_neighbors(query_vector, size, k, test)

    if k > 1:
        dst = []
        for i in range(len(neighbors)):
            for j in range(0, k-i):
                dst.append(neighbors[i][0])

        label = max(set(dst), key=dst.count)
    else:
        label = neighbors[0][0]



    return list_labels[int(label)]


def get_neighbors(query_vector, size, k, test):
    all_res = np.zeros((1, 2))
    all_res[0][1] = math.inf

    for i in range(cnt_labels):
        if test:
            matrix = test_matrices[i]
        else:
            matrix = all_mats[i]

        t = matrix.shape[0]
        dist_matrix = distance_mat(query_vector, matrix)
        name_mat = np.tile([i], (t, 1))
        result_label = np.hstack((name_mat, dist_matrix))
        all_res = np.vstack((all_res, result_label))

    sortedArr = all_res[(np.argsort(all_res[:, 1]))[:k]]

    return sortedArr


def learn_all(flag):
    res = []
    if flag:
        for k in range(5, 10):
            for size in range(15, 21):
                sum_percent = 0
                for validation in range(0, 5):
                    all_matrices(validation, size)
                    percent = validate(validation, size, k)
                    sum_percent += percent
                p = sum_percent/5
                # print("size:", size)
                # print("k:", k)
                # print("percent:", p)
                res.append([p, size, k])

    else:
        for size in range(15, 21):
            sum_percent = 0
            for validation in range(0, 5):
                all_matrices(validation, size)
                percent = validate(validation, size, 1)
                sum_percent += percent
            p = sum_percent / 5
            # print("size:", size)
            # print("k:", 1)
            # print("percent:", p)
            res.append([p, size, 1])

    return res


def test(size, k):
    get_test_matrices(size)
    corrects = 0
    m = 0

    for label in list_labels:
        good, n = check_directory_test(label, size, k)
        # print((good / n) * 100, "->", label)
        m += n
        corrects += good

    return (corrects / m) * 100


def check_directory_test(label, size, k):
    path = 'Data/Test/' + label

    dirListing = os.listdir(path)

    good = 0
    i = 0
    for filename in dirListing:
        dst = check_query(os.path.join(path, filename), size, k, True)
        if dst == label:
            good += 1
        i += 1

    return good, i


def get_test_matrices(size):
    global test_matrices
    test_matrices = []
    i = 0
    for label in list_labels:
        mat = make_mat(label, 0, size, True)
        test_matrices.append(mat)
        i += 1

    return test_matrices


##
# nearest neighbor(NN)
result = learn_all(False)
sortedArr = sorted(result, key=lambda x: x[0])

p, sizee, kk = sortedArr[0]
print(p, sizee, kk)

percent = test(sizee, kk)
print("k=1 ,percent test:", percent)


##
# k nearest neighbors (KNN)

result = learn_all(True)
sortedArr = sorted(result, key=lambda x: x[0])

p, sizee, kk = sortedArr[0]
print(p, sizee, kk)

percent = test(sizee, kk)
print("size is:", sizee)
print("k=",kk)
print("percent test:", percent)


##

