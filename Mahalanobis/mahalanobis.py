import numpy as np
from sklearn.datasets import load_iris

import itertools
import pandas as pd
import os

SAVE_CSV_PATH = "./csv/"
if not os.path.isdir(SAVE_CSV_PATH):
    os.makedirs(SAVE_CSV_PATH)

SAVE_TXT_PATH = "./txt/"
if not os.path.isdir(SAVE_TXT_PATH):
    os.makedirs(SAVE_TXT_PATH)

iris = load_iris()

feature_names = iris.feature_names
target_names = iris.target_names


def mahalanobis_dist(y_test, data):

    sigma = np.cov(data, rowvar=0, bias=0)

    sig_inv = np.linalg.inv(sigma)

    mu = np.average(data, axis=0)
    dif = y_test - mu

    dist = np.dot(dif, sig_inv)
    dist = np.dot(dist, dif)

    mah_dist = np.sqrt(dist)

    return mah_dist

def euclidian_dist(y_test, data, num_of_features):

    sigma = np.identity(num_of_features)
    sig_inv = np.linalg.inv(sigma)

    mu = np.average(data, axis=0)
    dif = y_test - mu

    dist = np.dot(dif, sig_inv)
    dist = np.dot(dist, dif)


    euc_dist = np.sqrt(dist)

    return euc_dist
    
def predict(x, y, z, label):

    pre = []

    for i in range(len(x)):
        count = 0
        tmp = x[i]
        if tmp > y[i]:
            tmp = y[i]
            count = 1
            if tmp > z[i]:
                count = 2

        elif tmp > z[i]:
            count = 2

        pre.append(count)
    
    accuracy = []
    miss = []

    for i in range(len(x)):
        if pre[i] == label[i]:
            accuracy.append(1)
        else:
            accuracy.append(0)
            miss.append(i)

    return np.average(accuracy), miss

def get_mahalanobis_csv(class1, class2, class3, y_test, file_name):

    mah_dist_list = []
    for i in range(y_test.shape[0]):
        mah_dist_list.append([mahalanobis_dist(y_test[i], class1),
                              mahalanobis_dist(y_test[i], class2),
                              mahalanobis_dist(y_test[i], class3)])

    df_mah = pd.DataFrame(mah_dist_list,
                          columns=[target_names[0], target_names[1], target_names[2]])


    print("Saving ==>> {}".format(file_name))
    df_mah.to_csv(file_name)

    return df_mah


def get_euclidian_csv(class1, class2, class3, y_test, file_name):

    euc_dist_list = []
    for i in range(y_test.shape[0]):
        euc_dist_list.append([euclidian_dist(y_test[i], class1, 2),
                              euclidian_dist(y_test[i], class2, 2),
                              euclidian_dist(y_test[i], class3, 2)])

    df_euc = pd.DataFrame(euc_dist_list,
                          columns=[target_names[0], target_names[1], target_names[2]])

    print("Saving ==>> {}".format(file_name))
    df_euc.to_csv(file_name)

    return df_euc

def get_result_txt(class1, class2, class3, y, label, file_name):

    acc, miss = predict(class1, class2, class3, label)

    print("Saving ==>> {}".format(file_name))

    with open(file_name, mode='w') as f:
        f.write("accuracy: {}\n\n".format(acc))

    if miss != []:

        with open(file_name, mode='a') as f:
            for i in range(len(miss)):
                f.write("miss data No.{}: {}\n".format(miss[i], y[miss[i]]))
                f.write("setosa    : {}\n".format(class1[miss[i]]))
                f.write("versicolor: {}\n".format(class2[miss[i]]))
                f.write("virginica : {}\n\n".format(class3[miss[i]]))


def main():

    for i, (a, b) in enumerate(itertools.combinations(range(4), 2)):
        d_index = [a, b]

        data = iris.data
        data = data[:, d_index]

        label = iris.target

        c1 = data[:40]
        c2 = data[50:90]
        c3 = data[100:140]

        y1 = data[40:50]
        y2 = data[90:100]
        y3 = data[140:150]
        y = np.concatenate((y1, y2, y3), axis=0)

        l1 = label[40:50]
        l2 = label[90:100]
        l3 = label[140:150]
        l = np.concatenate((l1, l2, l3), axis=0)

        file_name = os.path.join(SAVE_CSV_PATH,
                                 "mahalanobis_{}-{}.csv".format(feature_names[a], feature_names[b]))
        maha_pddf = get_mahalanobis_csv(c1, c2, c3, y, file_name)

        file_name = os.path.join(SAVE_CSV_PATH,
                                 "euclidian_{}-{}.csv".format(feature_names[a], feature_names[b]))
        eucl_pddf = get_euclidian_csv(c1, c2, c3, y, file_name)

        mah1 = maha_pddf[target_names[0]]
        mah2 = maha_pddf[target_names[1]]
        mah3 = maha_pddf[target_names[2]]

        euc1 = eucl_pddf[target_names[0]]
        euc2 = eucl_pddf[target_names[1]]
        euc3 = eucl_pddf[target_names[2]]

        file_name = os.path.join(SAVE_TXT_PATH,
                                 "mahalanobis_{}-{}.txt".format(feature_names[a], feature_names[b]))

        get_result_txt(mah1, mah2, mah3, y, l, file_name)

        file_name = os.path.join(SAVE_TXT_PATH,
                                 "euclidian_{}-{}.txt".format(feature_names[a], feature_names[b]))

        get_result_txt(euc1, euc2, euc3, y, l, file_name)


if __name__ == '__main__':
    main()

