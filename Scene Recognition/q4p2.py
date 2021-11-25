##
import cv2
import numpy as np
from sklearn.cluster import KMeans
import os.path

##
list_labels = ["Bedroom", "Coast", "Forest", "Highway", "Industrial", "Inside_City", "Kitchen", "Livingroom",
               "Mountain", "Office", "Open_Country", "Store", "Street", "Suburb", "Tall_Building"]
cnt_labels = len(list_labels)

num_des = 15
length = 2985

label_words = []
list_words = []

label_hist = []
list_hists = []


def load_all_words():
    global label_words
    global list_words

    label_words = []
    list_words = []

    label_cnt = -1

    for label in list_labels:
        label_cnt += 1
        path = 'Data/Train/' + label
        dirListing = os.listdir(path)
        for filename in dirListing:
            image = cv2.imread(os.path.join(path, filename))
            des = get_descriptors(image)

            label_words.append(label_cnt)
            list_words.append(des)

    return label_words


def make_all_hists(labels, k, centers):
    global label_hist
    global list_hists

    label_hist = []
    list_hists = []

    for i in range(length):
        l = label_words[i]
        des = list_words[i]
        hist = make_histogram(des,k,centers)

        label_hist.append(l)
        list_hists.append(np.array(hist))

    return label_hist


def get_descriptors(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)

    return des


def apply_kmeans(k):
    array = []
    for i in range(length):
        array.extend(list_words[i][0:num_des])

    array = np.array(array)

    kmeans = KMeans(n_clusters=k, random_state=0).fit(array)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    return labels, centers


def vector_distance(vector, mat):
    d = mat[:, :].astype(np.float64) - vector.astype(np.float64)
    d = np.absolute(d)
    dist = np.sum(d, axis=1).astype(np.float64)
    s = mat.shape[0]

    return dist.reshape(s, 1)


def make_histogram(des, k, centers):
    hist = [0] * k

    for vect in des:
        d = vector_distance(vect, centers)
        index = np.argmin(d)
        hist[index] += 1

    return hist


def check(centers, knn, k):
    number = 0
    good_cnt = 0

    for label in list_labels:
        good, n = check_directory(label, knn, k, centers)
        good_cnt += good
        number += n
        # print("Dir "+label+":", (good / n) * 100)

    return (good_cnt / number) * 100


def check_directory(label, knn, k, centers):
    path = 'Data/Test/' + label
    dirListing = os.listdir(path)

    good = 0
    n = 0
    for filename in dirListing:
        dst = check_query(os.path.join(path, filename), knn, k, centers)
        if dst == label:
            good += 1

        n += 1

    return good, n


def check_query(path, knn, k, centers):
    im = cv2.imread(path)
    des = get_descriptors(im)
    query_hist = make_histogram(des, k, centers)

    neighbors = get_neighbors(np.array(query_hist), knn)

    if knn > 1:
        dst = []
        for i in range(len(neighbors)):
            for j in range(0, knn-i):
                dst.append(neighbors[i][0])

        label = max(set(dst), key=dst.count)

    else:
        label = neighbors[0][0]

    return list_labels[int(label)]


def get_neighbors(query_hist, knn):
    matrix = np.array(list_hists)


    dist_matrix = vector_distance(query_hist, matrix)
    names = np.array(label_hist).reshape(length,1)
    all_res = np.hstack((names, dist_matrix))

    sortedArr = all_res[(np.argsort(all_res[:, 1]))[:knn]]

    return sortedArr


def test(k, knn):
    load_all_words()

    labels, centers = apply_kmeans(k)
    centers = np.array(centers)
    make_all_hists(labels, k, centers)
    percent = check(centers, knn, k)

    return percent


result = test(75, 8)
print("Test result is :", result, "percent")
##

