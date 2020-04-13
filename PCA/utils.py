import os

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

def get_distance(f):
    """Return the signed distance to the 0.5 levelset of a function."""

    # Prepare the embedding function.
    f = f > 0.5

    # Signed distance transform
    dist_func = ndimage.distance_transform_edt
    distance = np.where(f, -dist_func(f) + 0.5, (dist_func(1-f) - 0.5))

    return distance

def lsdm2label(f):

    f = f > 0.0

    label = np.where(f, 0, 1).astype(np.int)

    return label

def sigmoid(x, alpha=0.1):
    return 1.0 / (1.0 + np.exp(-alpha*x))  


def numpy2sitk(np_array):
    sitk_img = sitk.GetImageFromArray(np_array)
    return sitk_img

def sitk2numpy(sitk_img):
    np_array = sitk.GetArrayFromImage(sitk_img)  # (img(x,y,z)->numpyArray(z,y,x))
    return np_array

def read_mhd_and_raw(path):
    img = sitk.ReadImage(path)
    return img

def write_mhd_and_raw(Data, path):
    if not isinstance(Data, sitk.SimpleITK.Image):
        print('Please check your ''Data'' class')
        return False

    data_dir, file_name = os.path.split(path)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    sitk.WriteImage(Data, path, True)

    return True


def save_data_3D(path, img, num, size):
    """
    --------------------------------
    parameter
    --------------------------------
    path : save data directory
    img  : img array
    num  : number of images
    size : image size

    """
    if not os.path.isdir(path):
        os.mkdir(path)

    zero_f = num % 10

    for i in range(num):
        save_img = numpy2sitk(img[i].reshape(size, size, size))
        p = os.path.join(path, "output_" + str(i+1).zfill(zero_f+1) + ".mhd")
        write_mhd_and_raw(save_img, p)

