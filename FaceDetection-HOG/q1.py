##
import numpy as np
import shutil
import cv2
import os
import random
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import time
from joblib import dump, load
import pickle
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_precision_recall_curve
from sklearn import metrics
from skimage.feature import peak_local_max


def cut_and_resize():
    folder_name = 'validation'
    arr = os.listdir(folder_name)
    n = len(arr)

    for i in range(0, n):
        name = arr[i]
        if name == '.DS_Store':
            continue

        im = cv2.imread(folder_name + '/' + name)

        x, y, r = im.shape

        image = im[25:x-25, 25:y-25, :]
        resized = cv2.resize(image, (140, 140), interpolation=cv2.INTER_AREA)
        cv2.imwrite(folder_name + '/' + name, resized)

    return


def resize_folder(folder_name):
    arr = os.listdir(folder_name)
    n = len(arr)

    for i in range(0, n):
        name = arr[i]
        if name == '.DS_Store':
            continue

        im = cv2.imread(folder_name+'/'+name)
        resized = cv2.resize(im, (140, 140), interpolation = cv2.INTER_AREA)
        cv2.imwrite(folder_name+'/'+name, resized)

    return


def get_random_data(curr_folder, dst_folder, number):
    arr = os.listdir(curr_folder)
    n = len(arr)

    cnt = 0
    while True:
        r = random.randint(0, n - 1)

        dir_name = arr[r]
        if dir_name == '.DS_Store':
            continue
        in_dir = os.listdir(curr_folder + '/' + dir_name)

        if len(in_dir) > 0:
            r2 = random.randint(0, len(in_dir) - 1)
            st = in_dir[r2]
            original = r'' + str(curr_folder + '/' + arr[r] + '/' + st)
            target = r''+dst_folder+'/'
            shutil.move(original, target)
            cnt += 1

        if cnt >= number:
            break

    return


def get_hog(step):
    winSize, blockSize, blockStride, cellSize = step
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True
    HOG = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)
    return HOG


def get_all_vectors(win_Stride, Padding, hog):
    train = os.listdir('train')
    n1 = len(train)
    x = []
    y = []
    for i in range(0, n1):
        if train[i] == '.DS_Store':
            continue
        image = cv2.imread("train/" + train[i])
        vector = hog.compute(image, win_Stride, Padding)
        x.append(np.array(vector))
        y.append(1)

    train_false = os.listdir('negative-train')
    n0 = len(train_false)
    for i in range(0, n0):
        if train_false[i] == '.DS_Store':
            continue

        image = cv2.imread("negative-train/" + train_false[i])
        vector = hog.compute(image, win_Stride, Padding)

        x.append(np.array(vector))
        y.append(0)

    return x, y


def learn_parameters():
    first_percent = 0

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    # kernels = ['sigmoid']

    steps = [[(88, 136), (16, 16), (8, 8), (8, 8)],
             [(96, 136), (16, 16), (8, 8), (8, 8)],
             [(104, 136), (16, 16), (8, 8), (8, 8)],
             [(70, 120), (20, 20), (10, 10), (10, 10)],
             [(90, 120), (20, 20), (10, 10), (10, 10)],
             [(100, 130), (20, 20), (10, 10), (10, 10)],
             [(110, 110), (20, 20), (10, 10), (10, 10)],
             [(90, 130), (20, 20), (10, 10), (10, 10)]]

    list_all = []
    for step in steps:
        cp = time.time()

        winStride = (8 * step[3][0], 16 * step[3][1])

        print(winStride)
        hog = get_hog(step)
        X, Y = get_all_vectors(winStride, (0, 0), hog)

        x1 = np.array(X)
        nsamples, nx, ny = x1.shape
        d2 = x1.reshape((nsamples, nx * ny))

        for ker in kernels:
            CLF = multi_SVM(d2, Y, ker)
            tp, fp, tn, fn, all_cnt = valid(CLF, winStride,(0,0), hog)
            percent = 100 * (tp + tn) / all_cnt

            if percent > first_percent:
                first_percent = percent
                print("saving: ", step, "with: ", percent, ker)
                save_parameters(CLF, step)

            print(percent, step, winStride)
            list_all.append([percent, step, winStride])

        print("one loop finished in: ", time.time() - cp, " seconds")

    list_all = np.array(list_all)
    max_index = np.argmax(list_all[:, 0])
    step = list_all[max_index][1]
    print("best parameters are :", step, "with percent", list_all[max_index][0])

    return


def multi_SVM(X, Y, ker):
    clf = make_pipeline(StandardScaler(), SVC(kernel=ker, gamma='auto'))
    clf.fit(X, Y)
    return clf


def svm_predict(clf, query_vector):
    index = clf.predict(query_vector)

    if index[0] == 0:
        return False

    return True


def valid(clf, winStride1, padding, hog):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    cnt = 0

    pvalid = os.listdir('validation')
    s1 = len(pvalid)
    for i in range(0, s1):
        name = pvalid[i]
        if name == '.DS_Store':
            continue
        image = cv2.imread('validation/' + name)
        vector = hog.compute(image, winStride=winStride1, padding=(0, 0))

        t = len(vector)
        flag = svm_predict(clf, np.array(vector).reshape(1, t))
        if flag:
            tp += 1
        else:
            fp += 1

        cnt += 1

    nvalid = os.listdir('negative-validation')
    s2 = len(nvalid)
    for i in range(0, s2):
        name = nvalid[i]
        if name == '.DS_Store':
            continue
        image = cv2.imread('negative-validation/' + name)
        vector = hog.compute(image, winStride=winStride1, padding=(0,0))

        t = len(vector)
        flag = svm_predict(clf, np.array(vector).reshape(1, t))
        if flag:
            fn += 1
        else:
            tn += 1

        cnt += 1

    return tp, fp, tn, fn, cnt


def save_parameters(clf, step):
    dump(clf, 'clf.joblib')

    with open("learned.txt", "wb") as fp:
        pickle.dump(step, fp)

    print("saved")
    return


def load_parameters():
    with open("learned.txt", "rb") as fp:
        step = pickle.load(fp)

    clf = load('clf.joblib')

    return clf, step


def test(clf, step):
    y_test = []
    x_test = []
    y_score = []

    hog = get_hog(step)
    winStride1 = (8 * step[3][0], 16 * step[3][1])

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    cnt = 0

    ptest = os.listdir('test')
    s1 = len(ptest)
    for i in range(0, s1):
        name = ptest[i]
        if name == '.DS_Store':
            continue
        image = cv2.imread('test/' + name)
        vector = hog.compute(image, winStride=winStride1, padding=(0, 0))
        x_test.append(vector)
        y_test.append(1)

        t = len(vector)
        flag = svm_predict(clf, np.array(vector).reshape(1, t))
        if flag:
            tp += 1
            y_score.append(1)
        else:
            fp += 1
            y_score.append(0)
        cnt += 1

    ntest = os.listdir('negative-test')
    s2 = len(ntest)
    for i in range(0, s2):
        name = ntest[i]
        if name == '.DS_Store':
            continue
        image = cv2.imread('negative-test/' + name)
        vector = hog.compute(image, winStride=winStride1, padding=(0, 0))
        y_test.append(0)
        x_test.append(vector)

        t = len(vector)
        flag = svm_predict(clf, np.array(vector).reshape(1, t))
        if flag:
            fn += 1
            y_score.append(1)
        else:
            tn += 1
            y_score.append(0)

        cnt += 1

    percent = 100 * (tp + tn) / cnt

    average_precision = average_precision_score(y_test, y_score)

    X_test = np.array(x_test)
    nsamples, nx, ny = X_test.shape
    X_test = X_test.reshape((nsamples, nx * ny))
    disp = plot_precision_recall_curve(clf, X_test, y_test)
    disp.ax_.set_title('Precision-Recall: '
                       'AP={0:0.5f}'.format(average_precision))
    plt.show()

    sec = metrics.plot_roc_curve(clf, X_test, y_test)
    sec.ax_.set_title('RoC')
    plt.show()

    return percent, average_precision


def FaceDetector(input, thresh=0.4, dist=20, frame=140):
    clf, step = load_parameters()
    hog = get_hog(step)
    winStride = (8 * step[3][0], 16 * step[3][1])

    mat = np.zeros((input.shape[0] + 100, im.shape[1] + 100, 3))
    mat[50:im.shape[0] + 50, 50:im.shape[1] + 50, :] = input
    input = mat.astype(np.uint8)

    x_l, y_l, colors = input.shape

    values = np.ones((input.shape[0], input.shape[1]))
    values = values*(-4)

    # sliding window
    n = frame
    for x in range(0, x_l, 10):
        for y in range(0, y_l, 10):

            image = input[x:x+n, y:y+n, :]
            image = cv2.resize(image, (140, 140), interpolation=cv2.INTER_AREA)

            if (image.shape[0] != 140) | (image.shape[1] != 140):
                continue

            vector = hog.compute(image, winStride=winStride, padding=(5, 5))

            t = len(vector)
            # face = svm_predict(clf, np.array(vector).reshape(1, t))
            dec = clf.decision_function(np.array(vector).reshape(1, t))

            if dec < -1:
                values[x][y] = -2
            else:
                values[x][y] = dec

    print("finding local maxima")
    coordinates = peak_local_max(values, min_distance=dist, threshold_abs=thresh)
    print(coordinates)

    for point in coordinates:
        x, y = point
        cv2.rectangle(input, (y, x), (y + n, x + n), (255, 0, 0), 3)

    return input


##
# get random images from positive and negative images and move to folders

# get_random_data('lfw', 'train', 10000)
# get_random_data('lfw', 'test', 1000)
# get_random_data('lfw', 'validation', 1000)
#
# get_random_data('256_ObjectCategories', 'negative-train', 10000)
# get_random_data('256_ObjectCategories', 'negative-test', 1000)
# get_random_data('256_ObjectCategories', 'negative-validation', 1000)


# cut_and_resize()
# resize_folder('negative-train')
# resize_folder('negative-test')
# resize_folder('negative-validation')

##
# learn

learn_parameters()
print("executed")


##
# test

c, s = load_parameters()
print("loaded: ", s)
p, ap = test(c, s)
print("TEST RESULT: ", p, ap)


##
im = cv2.imread('Melli.jpg')
result = FaceDetector(im, 0.55, 70, 140)
cv2.imwrite('res4.jpg', result)


##
im = cv2.imread('Persepolis.jpg')
result = FaceDetector(im, 0.82, 10, 140)
cv2.imwrite('res5.jpg', result)


##
im = cv2.imread('Esteghlal.jpg')
result = FaceDetector(im, 0.8, 10, 270)
cv2.imwrite('res6.jpg', result)


