import pandas as pd
import matplotlib.pyplot as plt

import os

SAVE_IMAGE_PATH = "./images/"
if not os.path.isdir(SAVE_IMAGE_PATH):
    os.makedirs(SAVE_IMAGE_PATH)

def load_csv(path):
    df = pd.read_csv(path)
    return df

def plot_gen(df, case, file_name):
    plt.figure()
    plt.xlabel("case number")
    plt.ylabel("generalization")
    plt.ylim(0, 1)

    left = case
    height = df[:-1]["Generalization"]

    plt.bar(left, height)
    plt.grid()

    print("Saving ==>> {}".format(file_name))
    plt.savefig(file_name)

def plot_spe(df, n_arbit, file_name):

    spe = []
    t_spe = 0

    for i in range(n_arbit):
        t_spe += df.at[i, 'Specificity']
        spe.append(t_spe/(i+1))

    plt.figure()
    plt.xlabel("number of shapes")
    plt.ylabel("specificity")
    plt.ylim(0, 1)

    plt.plot(list(range(1, n_arbit+1)), spe)
    plt.grid()

    print("Saving ==>> {}".format(file_name))
    plt.savefig(file_name)

def plot_gen_and_spe(n_component, gen, spe, file_name):
    plt.figure()
    plt.xlabel("number of components")
    plt.ylabel("rate")

    plt.plot(list(range(1, n_component+1)), gen, label="gen")
    plt.plot(list(range(1, n_component+1)), spe, label="spe")

    plt.grid()
    plt.legend()

    print("Saving ==>> {}".format(file_name))
    plt.savefig(file_name)

def main():
    
    component = ["component={}.csv".format(i+1) for i in range(9)]

    case1 = ["{}".format(i+1) for i in range(10)]
    case2 = ["{}".format(i+11) for i in range(10)]

    df = pd.read_csv("./csv/check/1fold/component=9.csv")
    plot_gen(df, case1, os.path.join(SAVE_IMAGE_PATH, "train_gen.png"))

    df = pd.read_csv("./csv/gen/1fold/component=9.csv")
    plot_gen(df, case2, os.path.join(SAVE_IMAGE_PATH, "test_gen.png"))

    df = pd.read_csv("./csv/spe/1fold/component=9.csv")
    plot_spe(df, n_arbit=100, file_name=os.path.join(SAVE_IMAGE_PATH, "spe.png"))

    gen_total_1st, spe_total_1st = [], []
    gen_total_2nd, spe_total_2nd = [], []
    for co in component:
        df_gen_1st = pd.read_csv(os.path.join("./csv/gen/1fold/", co), index_col=0)
        df_spe_1st = pd.read_csv(os.path.join("./csv/spe/1fold/", co), index_col=0)

        df_gen_2nd = pd.read_csv(os.path.join("./csv/gen/2fold/", co), index_col=0)
        df_spe_2nd = pd.read_csv(os.path.join("./csv/spe/2fold/", co), index_col=0)
        
        gen_total_1st.append(df_gen_1st.at['total', "Generalization"])
        spe_total_1st.append(df_spe_1st.at['total', "Specificity"])

        gen_total_2nd.append(df_gen_2nd.at['total', "Generalization"])
        spe_total_2nd.append(df_spe_2nd.at['total', "Specificity"])
    
    plot_gen_and_spe(9, gen_total_1st, spe_total_1st, os.path.join(SAVE_IMAGE_PATH, "1st_eigen_space.png"))
    plot_gen_and_spe(9, gen_total_2nd, spe_total_2nd, os.path.join(SAVE_IMAGE_PATH, "2nd_eigen_space.png"))


if __name__ == "__main__":
    main()

    
