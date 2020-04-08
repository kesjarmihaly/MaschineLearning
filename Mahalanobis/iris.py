import itertools 

import matplotlib.pyplot as plt

from sklearn import datasets
import numpy as np

import os

SAVE_IMAGE_PATH = "./images/"
if not os.path.isdir(SAVE_IMAGE_PATH):
    os.makedirs(SAVE_IMAGE_PATH)

def get_data():
    iris_datasets = datasets.load_iris()
    print(iris_datasets.DESCR)

    return iris_datasets

def train_data(x_data):

    feature = x_data.data
    feature_names = x_data.feature_names
    target_names = x_data.target_names

    c1 = feature[:40]
    c2 = feature[50:90]
    c3 = feature[100:140]

    features = np.concatenate((c1, c2, c3), axis=0)

    label = x_data.target

    l1 = label[:40]
    l2 = label[50:90]
    l3 = label[100:140]

    targets = np.concatenate((l1, l2, l3), axis=0)

    plt.figure(figsize=(20, 12))

    for i, (x, y) in enumerate(itertools.combinations(range(4), 2)):
        plt.subplot(2, 3, i + 1)


        for t, marker, c in zip(range(3), '>ox', 'rgb'):
            plt.scatter(
                features[targets == t, x],
                features[targets == t, y],
                marker=marker,
                c=c,
                label=target_names[t]
            )
            plt.xlabel(feature_names[x])
            plt.ylabel(feature_names[y])
            plt.autoscale()
            plt.grid()
            plt.legend()

    image_path = os.path.join(SAVE_IMAGE_PATH, "train_feature.png")
    print("Saving ==>> {}".format(image_path))
    plt.savefig(image_path)


def test_data(t_data):

    data = t_data.data
    feature_names = t_data.feature_names
    target_names = t_data.target_names

    c1 = data[40:50]
    c2 = data[90:100]
    c3 = data[140:150]

    features = np.concatenate((c1, c2, c3), axis=0)

    label = t_data.target

    l1 = label[40:50]
    l2 = label[90:100]
    l3 = label[140:150]

    targets = np.concatenate((l1, l2, l3), axis=0)

    plt.figure(figsize=(20, 12))

    for i, (x, y) in enumerate(itertools.combinations(range(4), 2)):
        plt.subplot(2, 3, i + 1)

        for t, marker, c in zip(range(3), '>ox', 'rgb'):
            plt.scatter(
                features[targets == t, x],
                features[targets == t, y],
                marker=marker,
                c=c,
                label=target_names[t]
            )
            plt.xlabel(feature_names[x])
            plt.ylabel(feature_names[y])
            plt.autoscale()
            plt.grid()
            plt.legend()

    image_path = os.path.join(SAVE_IMAGE_PATH, "test_feature.png")
    print("Saving ==>> {}".format(image_path))
    plt.savefig(image_path)


def main():

    iris = get_data()
    train_data(iris)
    test_data(iris)


if __name__ == '__main__':
    main()