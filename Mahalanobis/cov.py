import numpy as np
from matplotlib import pyplot as plt
import os

SAVE_IMAGE_PATH = "./images/"
if not os.path.isdir(SAVE_IMAGE_PATH):
    os.makedirs(SAVE_IMAGE_PATH)

def visualize_cov_matrix(mean, cov_list, title_list):

    for cov, title in zip(cov_list, title_list):

        x, y = np.random.multivariate_normal(mean, cov, 1000).T

        plt.figure()
        plt.plot(x, y, 'x')
        plt.title(title)
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)

        image_path = os.path.join(SAVE_IMAGE_PATH, "{}.png".format(title))

        print("Saving ==>> {}".format(image_path))
        plt.savefig(image_path)

def get_cov_matrix_list():

    covariance_component_list = [-1, -0.7, 0, 0.7, 1]

    cov_list = []
    title_list = []
    for c in covariance_component_list:
        cov = np.array([[1, c], [c, 1]])
        cov_list.append(cov)

        title = "covariance = {}, sigma_x = 1, sigma_y = 1".format(c)
        title_list.append(title)

    sigma_list = [1, 3, 5]
    for s in sigma_list:
        cov = np.array([[1, 0.7], [s, 1]])
        cov_list.append(cov)

        title = "covariance = 0.7, sigma_x = 1, sigma_y = {}".format(s)
        title_list.append(title)

    return cov_list, title_list


def main():

    mean = np.array([0, 0])

    cov_list, title_list = get_cov_matrix_list()

    visualize_cov_matrix(mean, cov_list, title_list)


if __name__ == '__main__':
    main()