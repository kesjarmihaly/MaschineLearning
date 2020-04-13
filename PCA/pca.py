import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import utils
import glob

import model

DATA_PATH = "./data/"

SAVE_ORIGIN_PATH = "./origin/"
SAVE_NPY_PATH = "./npy/"
SAVE_OUTPUT_PATH = "./output/"
SAVE_ARBIT_PATH = "./arbit/"
SAVE_SHAPE_CHAGE_PATH = "./shape_change/"
SAVE_IMAGE_PATH = "./images/"

if not os.path.isdir(SAVE_ORIGIN_PATH):
    os.makedirs(SAVE_ORIGIN_PATH)
if not os.path.isdir(SAVE_NPY_PATH):
    os.makedirs(SAVE_NPY_PATH)
if not os.path.isdir(SAVE_OUTPUT_PATH):
    os.makedirs(SAVE_OUTPUT_PATH)
if not os.path.isdir(SAVE_ARBIT_PATH):
    os.makedirs(SAVE_ARBIT_PATH)
if not os.path.isdir(SAVE_SHAPE_CHAGE_PATH):
    os.makedirs(SAVE_SHAPE_CHAGE_PATH)
if not os.path.isdir(SAVE_IMAGE_PATH):
    os.makedirs(SAVE_IMAGE_PATH)

def check_path(p):
    if not os.path.isdir(p):
        os.makedirs(p)


def load_data():

    load_file_list = glob.glob(os.path.join(DATA_PATH, "*.mhd"))

    images = []
    for f in load_file_list:
        img = utils.read_mhd_and_raw(f)
        img = utils.sitk2numpy(img)
        images.append(img)
   
    image_shape = images[0].shape
    np_images = np.asarray(images).reshape(len(images), -1)

    return np_images, image_shape


def save_image(images, img_shape, output_dir, name_list=None, num_arbit=False):
    check_path(output_dir)

    if name_list is not None:
        for i, name in enumerate(name_list):
            np_img = utils.lsdm2label(images[i]).reshape(img_shape)
            sitk_img = utils.numpy2sitk(np_img)

            image_file = os.path.join(output_dir, name)
            print("Saving ==>> {}".format(image_file))
            utils.write_mhd_and_raw(sitk_img, image_file)

    elif num_arbit:

        np_img = utils.lsdm2label(images).reshape(img_shape)
        sitk_img = utils.numpy2sitk(np_img)

        image_file = os.path.join(output_dir, "output_"+str(num_arbit)+".mhd")
        print("Saving ==>> {}".format(image_file))
        utils.write_mhd_and_raw(sitk_img, image_file)

    else:
        for i in range(images.shape[0]):
            np_img = utils.lsdm2label(images[i]).reshape(img_shape)
            sitk_img = utils.numpy2sitk(np_img)

            image_file = os.path.join(output_dir, "output_" + str(i+1).zfill(2) + ".mhd")
            print("Saving ==>> {}".format(image_file))
            utils.write_mhd_and_raw(sitk_img, image_file)


def get_cr_and_ccr_fig(df, save_dir):
    plt.figure()
    plt.xlabel("n_component")
    plt.ylabel("rate")
    plt.plot(df["軸"][:-2], df["寄与率"][:-2], label="cr")
    plt.plot(df["軸"][:-2], df["累積寄与率"][:-2], label="ccr")
    plt.grid()
    plt.legend()

    file_name = os.path.join(save_dir, "cr_and_ccr.png")
    print("Saving ==>> {}".format(file_name))
    plt.savefig(file_name)


def get_cr_and_ccr_csv(eigen_val, pca_cr, pca_ccr, save_dir):
    check_path(save_dir)

    cr_and_ccr_list = []
    for i, (val, cr, ccr) in enumerate(zip(eigen_val, pca_cr, pca_ccr)):
        cr_and_ccr_list.append([i+1, val, cr, ccr])

    columns = ["軸", "固有値", "寄与率", "累積寄与率"]
    df = pd.DataFrame(cr_and_ccr_list, columns=columns)

    file_name = os.path.join(save_dir, "cr_and_ccr.csv")
    print("Saving ==>> {}".format(file_name))
    df.to_csv(file_name, index=False)

    get_cr_and_ccr_fig(df, save_dir)


def process_train(n_component, x_train, save_dir):
    
    pca = model.ProcessPCA(n_component)
    
    pca.fit(x_train)

    pca_mean = pca.get_mean_vect()
    pca_eigen_val = pca.get_eigen_value()
    pca_eigen_vec = pca.get_eigen_vect()

    pca_cr = pca.get_cr()
    pca_ccr = pca.get_ccr()

    check_path(save_dir)

    if n_component == x_train.shape[0] - 1:
        get_cr_and_ccr_csv(list(pca_eigen_val), pca_cr, pca_ccr, save_dir)
        
    np.save(os.path.join(save_dir, "mean_vect.npy"), pca_mean)
    np.save(os.path.join(save_dir, "eigen_value.npy"), pca_eigen_val)
    np.save(os.path.join(save_dir, "eigen_vect.npy"), pca_eigen_vec)


def process_test(n_component, y_test, save_dir):

    pca = model.ProcessPCA(n_component)

    mean_vect   = np.load(os.path.join(save_dir, "mean_vect.npy"))
    eigen_value = np.load(os.path.join(save_dir, "eigen_value.npy"))
    eigen_vect  = np.load(os.path.join(save_dir, "eigen_vect.npy"))

    pca.set_mean_vect(mean_vect)
    pca.set_eigen_value(eigen_value)
    pca.set_eigen_vect(eigen_vect)

    projected = pca.projection(y_test)
    reconstructed = pca.reconstruction(projected)

    return projected, reconstructed


def process_arbit(n_component, x_train, save_dir, n_shape, seed, img_shape, fold):

    np.random.seed(seed=seed)

    pca = model.ProcessPCA(n_component)

    mean_vect   = np.load(os.path.join(save_dir, "mean_vect.npy"))
    eigen_value = np.load(os.path.join(save_dir, "eigen_value.npy"))
    eigen_vect  = np.load(os.path.join(save_dir, "eigen_vect.npy"))

    pca.set_mean_vect(mean_vect)
    pca.set_eigen_value(eigen_value)
    pca.set_eigen_vect(eigen_vect)

    projected = pca.projection(x_train)

    loc = 0
    scale = eigen_value

    z_sample = np.random.normal(loc=loc, scale=np.sqrt(scale[0]), size=[n_shape, 1])
    for i in range(n_component - 1):
        temp = np.random.normal(loc=loc, scale=np.sqrt(scale[i + 1]), size=[n_shape, 1])
        z_sample = np.concatenate([z_sample, temp], axis=1)

    for i in range(n_shape):
        arb = pca.reconstruction(z_sample[i])
        save_image(arb, img_shape, os.path.join(SAVE_ARBIT_PATH, fold), num_arbit=(i+1))


def process_shape_change(save_dir):

    pca = model.ProcessPCA(n_component=2)

    mean_vect   = np.load(os.path.join(save_dir, "mean_vect.npy"))
    eigen_value = np.load(os.path.join(save_dir, "eigen_value.npy"))
    eigen_vect  = np.load(os.path.join(save_dir, "eigen_vect.npy"))

    pca.set_mean_vect(mean_vect)
    pca.set_eigen_value(eigen_value)
    pca.set_eigen_vect(eigen_vect)

    grid_x = np.linspace(-3, 3, 7)
    grid_y = np.linspace(-3, 3, 7)[::-1]

    name_list = []
    samples = []
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            xi *= np.sqrt(eigen_value[0])
            yi *= np.sqrt(eigen_value[1])
            z_sample = np.array([[xi, yi]])
            name_list.append("z_{}sigmaX_{}sigmaY.mhd".format(i-3, j-3))
            samples.append(pca.reconstruction(z_sample))

    return samples, name_list


def visual_latent_space(projected, save_dir):

    plt.figure()
    plt.scatter(projected[:, 0], projected[:, 1])
    plt.grid()

    check_path(save_dir)
    file_name = os.path.join(save_dir, "latent.png")
    print("Saving ==>> {}".format(file_name))
    plt.savefig(file_name)


def two_fold_cross_validation(n_component, data, img_shape):

    seed = 1
    n_shape = 100

    one_fold = os.path.join("1fold", "component={}".format(n_component))
    two_fold = os.path.join("2fold", "component={}".format(n_component))

    for i in range(2):
        if i == 0:
            x_train, y_test = data[:10], data[10:]
            process_train(n_component, x_train, os.path.join(SAVE_NPY_PATH, one_fold))

            proj, rec = process_test(n_component, y_test, os.path.join(SAVE_NPY_PATH, one_fold))

            save_image(rec, img_shape, os.path.join(SAVE_OUTPUT_PATH, one_fold))

            process_arbit(n_component, x_train, os.path.join(SAVE_NPY_PATH, one_fold), n_shape, seed, img_shape, one_fold)


        else:
            x_train, y_test = data[10:], data[:10]
            process_train(n_component, x_train, os.path.join(SAVE_NPY_PATH, two_fold))

            proj, rec = process_test(n_component, y_test, os.path.join(SAVE_NPY_PATH, two_fold))
            save_image(rec, img_shape, os.path.join(SAVE_OUTPUT_PATH, two_fold))

            process_arbit(n_component, x_train, os.path.join(SAVE_NPY_PATH, one_fold), n_shape, seed, img_shape, two_fold)
            

def train(data, img_shape):

    total_component = 19 # 出力は第20軸まで出てしまいますが、実質、第19軸までです。

    process_train(total_component, data, SAVE_NPY_PATH)


    n_component = 2
    proj, rec = process_test(n_component, data, SAVE_NPY_PATH)

    visual_latent_space(proj, SAVE_IMAGE_PATH)
    samp, na = process_shape_change(SAVE_NPY_PATH)
    save_image(samp, img_shape, os.path.join(SAVE_SHAPE_CHAGE_PATH), name_list=na)

    n_component = 9 # 出力は第10軸まで出てしまいますが、実質、第9軸までです。
    x_train, y_test = data[:10], data[:10]
    process_train(n_component, x_train, os.path.join(SAVE_NPY_PATH, "check"))
    proj, rec = process_test(n_component, y_test, os.path.join(SAVE_NPY_PATH, "check"))
    save_image(rec, img_shape, os.path.join(SAVE_OUTPUT_PATH, "check"))


def main():

    data, img_shape = load_data()
    save_image(data, img_shape, SAVE_ORIGIN_PATH)

    train(data, img_shape)

    total_component = 9
    for n_component in range(total_component):
        two_fold_cross_validation(n_component+1, data, img_shape)


if __name__ == '__main__':
    main()
            