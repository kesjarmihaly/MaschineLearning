import numpy as np
from sklearn.metrics import jaccard_score
import pandas as pd
import torch

import utils
import glob
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


SAVE_CSV_PATH = "./csv/"
if not os.path.isdir(SAVE_CSV_PATH):
    os.makedirs(SAVE_CSV_PATH)

def jaccard_index(y_true, y_pred):

    y_true = torch.from_numpy(y_true)
    y_pred = torch.from_numpy(y_pred)

    AandB = torch.sum(y_true & y_pred)
    AorB =torch.sum(y_true | y_pred)

    jc_idx = AandB.numpy()/AorB.numpy()

    return jc_idx


def Generalization(y_true, y_pred):
    
    gen = []
    for y_t, y_p in zip(y_true, y_pred):
        gen.append(jaccard_index(y_t, y_p))
    
    t_gen = sum(gen)/len(gen)

    return gen, t_gen

def Specificity(y_true, y_pred):

    spe = []
    for i, y_p in enumerate(y_pred):
        jac = []
        for y_t in y_true:
            jac.append(jaccard_index(y_t, y_p))
        spe.append(max(jac))
    
    t_spe = sum(spe)/len(spe)

    return spe, t_spe

def load_data(data_path):

    load_file_list = glob.glob(os.path.join(data_path, "*.mhd"))

    images = []
    for f in load_file_list:
        img = utils.read_mhd_and_raw(f)
        img = utils.sitk2numpy(img)
        images.append(img)
   
    image_shape = images[0].shape
    np_images = np.asarray(images).reshape(len(images), -1)

    return np_images, image_shape

def check_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p)

def get_csv(file_name, data, index=None, columns=None):

    df = pd.DataFrame(data, index=index, columns=columns)

    print("Saving ==>> {}".format(file_name))
    df.to_csv(file_name)

def main():

    folds = ["1fold", "2fold"]
    check = "check"

    index_gen = ["{}".format(i+1) for i in range(10)]
    index_gen.append("total")

    n_shape = 100
    index_spe = ["{}".format(i+1) for i in range(100)]
    index_spe.append("total")

    column_gen = ["Generalization"]
    column_spe = ["Specificity"]

    SAVE_ORIGIN_PATH = "./origin/"
    SAVE_OUTPUT_PATH = "./output/"
    SAVE_ARBIT_PATH = "./arbit/"

    origin_img, _ = load_data(SAVE_ORIGIN_PATH)

    component = ["component={}".format(i+1) for i in range(9)]

    # for fo in folds:

    #     if fo == "1fold":
    #         y_true = origin_img[10:]
    #     else:
    #         y_true = origin_img[:10]

    #     for co in component:
    #         y_pred, _ = load_data(os.path.join(SAVE_OUTPUT_PATH, os.path.join(fo, co)))
    #         gen, t_gen = Generalization(y_true, y_pred)
    #         gen.append(t_gen)

    #         save_dir = os.path.join(SAVE_CSV_PATH, "gen" + "/" + fo + "/")
    #         check_dir(save_dir)

    #         file_name = os.path.join(save_dir, co + ".csv")
    #         get_csv(file_name, gen, index, column_gen)
    
    #     if fo == "1fold":
    #         y_true = origin_img[:10]
    #         y_pred, _ = load_data(os.path.join(SAVE_OUTPUT_PATH, check))
    #         gen, t_gen = Generalization(y_true, y_pred)
    #         gen.append(t_gen)

    #         save_dir = os.path.join(SAVE_CSV_PATH, check + "/" + fo + "/")
    #         check_dir(save_dir)

    #         file_name = os.path.join(save_dir, "component=9" + ".csv")
    #         get_csv(file_name, gen, index, column_gen)
        
    
    for fo in folds:
        if fo == "1fold":
            y_true = origin_img[10:]
        else:
            y_true = origin_img[:10]

        for co in component:
            y_pred, _ = load_data(os.path.join(SAVE_ARBIT_PATH, os.path.join(fo, co)))
            spe, t_spe = Specificity(y_true, y_pred)
            spe.append(t_spe)

            save_dir = os.path.join(SAVE_CSV_PATH, "spe" + "/" + fo + "/")
            check_dir(save_dir)

            file_name = os.path.join(save_dir, co + ".csv")
            get_csv(file_name, spe, index_spe, column_spe)


if __name__ == "__main__": 
    main()