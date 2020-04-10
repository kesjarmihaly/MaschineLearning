import numpy as np

from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import seaborn as sns

import os

SAVE_IMAGE_PATH = "./images/"
if not os.path.isdir(SAVE_IMAGE_PATH):
    os.makedirs(SAVE_IMAGE_PATH)

SAVE_CSV_PATH = "./csv/"
if not os.path.isdir(SAVE_IMAGE_PATH):
    os.makedirs(SAVE_IMAGE_PATH)

iris = load_iris()

feature_names = iris.feature_names
target_names = iris.target_names


def plot_decision_regions(k, acc, x_train, label_train, y_test=None, label_test=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(label_train))])

    x1_min, x1_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    x2_min, x2_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    z = knn(k, np.array([xx1.ravel(), xx2.ravel()]).T, x_train, label_train)
    z = np.asarray(z)
    z = z.reshape(xx1.shape)

    plt.figure()

    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)

    plt.title("k={}, Accuracy: {:.3f}".format(k, acc))
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])

    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(label_train)):
        plt.scatter(x=x_train[label_train == cl, 0], y=x_train[label_train == cl, 1], alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=target_names[cl])
        plt.scatter(x=y_test[label_test == cl, 0], y=y_test[label_test == cl, 1], alpha=0.8, c=cmap(idx),
                    marker=markers[idx])

    if y_test is not None:
        x_test, y_test = y_test, label_test
        plt.scatter(x_test[:, 0], x_test[:, 1], c='', alpha=1, linewidths=1, marker='o', s=55, edgecolors="black",
                    label='test set')

    plt.legend()

    image_path = os.path.join(SAVE_IMAGE_PATH, "decision_region")
    if not os.path.isdir(image_path):
        os.makedirs(image_path)

    image_path = os.path.join(image_path, "decision_region-k={}.png".format(k))
    print("Saving ==>> {}".format(image_path))
    plt.savefig(image_path)


def load_data():

    data = iris.data
    label = iris.target

    x1 = data[:40]
    x2 = data[50:90]
    x3 = data[100:140]
    x_train = np.concatenate((x1, x2, x3), axis=0)

    ltr1 = label[:40]
    ltr2 = label[50:90]
    ltr3 = label[100:140]
    l_train = np.concatenate((ltr1, ltr2, ltr3), axis=0)

    y1 = data[40:50]
    y2 = data[90:100]
    y3 = data[140:150]
    y_test = np.concatenate((y1, y2, y3), axis=0)

    lte1 = label[40:50]
    lte2 = label[90:100]
    lte3 = label[140:150]
    l_test = np.concatenate((lte1, lte2, lte3), axis=0)

    return x_train, l_train, y_test, l_test

def get_csv(data_frame, file_name):
    print("Saving ==>> {}".format(file_name))
    data_frame.to_csv(file_name)

def l2_norm(x, y):
    l2 = np.square(x - y)
    return l2


def knn(k, y_test, x_train, label_train):
    distances = []


    for i in range(y_test.shape[0]):
        distances.append(np.sum(l2_norm(x_train, y_test[i]), axis=1))

    sorted_train_indexes = np.argsort(distances)

    sorted_k_labels = label_train[sorted_train_indexes][:, :k]

    labels = []
    for i in range(y_test.shape[0]):
        label = Counter(sorted_k_labels[i]).most_common(1)[0][0]
        labels.append(label)

    return labels

def predict(pred, test):
    return np.sum(pred == test) / len(test)

def get_confusion_matrix(k, y_test, y_pred):

    plt.figure()
    mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(mat, square=True, annot=True, cbar=False, fmt='d', cmap='RdPu')

    plt.title('k={}'.format(k))
    plt.xlabel('predicted class')
    plt.ylabel('true value')

    image_path = os.path.join(SAVE_IMAGE_PATH, "heatmap")
    if not os.path.isdir(image_path):
        os.makedirs(image_path)

    image_path = os.path.join(image_path, "heatmap-k={}.png".format(k+1))

    print("Saving ==>> {}".format(image_path))
    plt.savefig(image_path)


def main():

    k = 120

    x_train, label_train, y_test, label_test = load_data()

    x_train = x_train[:, :2]
    y_test = y_test[:, :2]

    acc_list = []

    for k_i in range(k):
        label_pred = knn(k_i+1, y_test, x_train, label_train)

        acc = predict(label_pred, label_test)
        acc_list.append([k_i+1, acc])

        plot_decision_regions(k_i+1, acc, x_train, label_train, y_test, label_test)
        get_confusion_matrix(k_i, label_test, label_pred)

    df = pd.DataFrame(data=acc_list, columns=["k", "Accuracy"])

    file_name = os.path.join(SAVE_CSV_PATH, "knn_accuracy.csv")
    get_csv(df, file_name)


if __name__ == '__main__':
    main()
