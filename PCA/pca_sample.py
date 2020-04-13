import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

import os

SAVE_IMAGE_PATH = "./images/"
if not os.path.isdir(SAVE_IMAGE_PATH):
    os.makedirs(SAVE_IMAGE_PATH)

 
def pca_use_org(data):

    cov_matrix = np.cov(data, rowvar=False)

    l, v = np.linalg.eig(cov_matrix)
 
    l_index = np.argsort(l)[::-1]
    l_ = l[l_index]
    v_ = v[:, l_index]
 
    data_trans = np.dot(data, v_)
 
    return data_trans, v_

def load_data():
    
    d_index = [0, 2]
    iris = load_iris()
    data = iris.data
    data = data[:, d_index]
    target = iris.target
    target_names = iris.target_names
    feature_names = iris.feature_names

    return data, target, target_names, feature_names

def plot_matlib(data, target, target_names, feature_names, data_trans, v):
    
    vec_s = [0, 0]
    vec_1st_e = [2*v[0, 0], 2*v[0, 1]]
    vec_2nd_e = [2*v[1, 0], 2*v[1, 1]]

    plt.figure(figsize=[8, 8])
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.quiver(vec_s[0], vec_s[1], vec_1st_e[0], vec_1st_e[1],
               angles='xy', scale_units='xy', scale=1, color='r', label='1st')
    plt.quiver(vec_s[0], vec_s[1], vec_2nd_e[0], vec_2nd_e[1],
               angles='xy', scale_units='xy', scale=1, color='b', label='2nd')

    for t, c in zip(range(3), 'rgb'):
        plt.scatter(data[target == t, 0], data[target == t, 1], c=c, label=target_names[t])
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[2])
        plt.grid()
        plt.legend()
    
    file_name = os.path.join(SAVE_IMAGE_PATH, "pca_sample_fig1.png")
    print("Saving ==>> {}".format(file_name))
    plt.savefig(file_name)
 
    plt.figure(figsize=[8, 8])
    plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    plt.title('1st principal - 2nd principal')

    for t, c in zip(range(3), 'rgb'):
        plt.scatter(data_trans[target == t, 0], data_trans[target == t, 1], c=c, label=target_names[t])
        plt.xlabel('1st principal')
        plt.ylabel('2nd principal')
        plt.grid()
        plt.legend()


    plt.subplot2grid((4, 1), (2, 0))
    plt.tick_params(labelleft="off", left="off")
    plt.title('1st principal')

    for t, c in zip(range(3), 'rgb'):
        plt.scatter(data_trans[target == t, 0], np.zeros(len(data_trans[target == t, 0])), c=c, label=target_names[t])
        plt.grid()

    plt.subplot2grid((4, 1), (3, 0))
    plt.title('2nd principal')
    plt.tick_params(labelleft="off", left="off")

    for t, c in zip(range(3), 'rgb'):
        plt.scatter(data_trans[target == t, 1], np.zeros(len(data_trans[target == t, 1])), c=c, label=target_names[t])
        plt.grid()

    plt.tight_layout()
    
    file_name = os.path.join(SAVE_IMAGE_PATH, "pca_sample_fig2.png")
    print("Saving ==>> {}".format(file_name))
    plt.savefig(file_name)

def main():

   data, target, target_names, feature_names = load_data()

   data = data - data.mean(axis=0)

   data_trans, v = pca_use_org(data)

   plot_matlib(data, target, target_names, feature_names, data_trans, v)
    
 
 
if __name__ == "__main__":
    main()
    