##
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
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


##
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


def check(centers, k, clf):
    number = 0
    good_cnt = 0

    confusion = np.zeros((cnt_labels, cnt_labels))

    for label in list_labels:
        good, n, res_list = check_directory(label, k, centers, clf)
        confusion = confusion + res_list
        good_cnt += good
        number += n
        # print("Dir "+label+":", (good / n) * 100)

    return (good_cnt / number) * 100, confusion


def check_directory(label, k, centers, clf):
    path = 'Data/Test/' + label
    dirListing = os.listdir(path)

    res_list = []
    y_true = []

    good = 0
    n = 0
    for filename in dirListing:
        dst = check_query(os.path.join(path, filename), k, centers, clf)
        res_list.append(dst)
        y_true.append(label)
        if dst == label:
            good += 1

        n += 1

    confusion_mat = confusion_matrix(y_true, res_list, list_labels)

    return good, n, confusion_mat


def check_query(path, k, centers, clf):
    im = cv2.imread(path)
    des = get_descriptors(im)
    query_hist = make_histogram(des, k, centers)
    query_hist = np.array(query_hist).reshape((1,k))
    svm_response = svm_predict(clf, query_hist)

    return svm_response


def multi_SVM():
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(list_hists, label_hist)
    return clf


def svm_predict(clf, query_hist):
    index = clf.predict(query_hist)

    return list_labels[index[0]]


def test(k):
    load_all_words()

    labels, centers = apply_kmeans(k)
    centers = np.array(centers)
    make_all_hists(labels, k, centers)

    clf = multi_SVM()

    percent, confusion = check(centers, k, clf)

    return percent, confusion


##
p, res_mat = test(100)

##
print("success ratio for k=100 is:", p)

##
df_cm = pd.DataFrame(res_mat, index=list_labels, columns=list_labels)
f, ax = plt.subplots(figsize=(10, 7))

ax = sn.heatmap(df_cm, annot=True)
ax.set_title('success ratio='+str(p))
plt.savefig('res09.jpg')



