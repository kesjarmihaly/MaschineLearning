import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

import os

SAVE_RESULT_PATH = "./result/"
if not os.path.isdir(SAVE_RESULT_PATH):
    os.makedirs(SAVE_RESULT_PATH)

iris = load_iris()

feature_names = iris.feature_names
target_names = iris.target_names

class NeuralNetwork:

    def __init__(self, eta=0.1):
        self.w = np.ones(4) / 10  # wの初期値は全部0.1
        self.b = np.ones(1) / 10  # bも初期値を0.1にする。
        self.eta = eta

    def get_weight(self):
        return self.w

    def get_bias(self):
        return self.b

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def activation(self, X):
        return self.sigmoid(np.dot(X, self.w) + self.b)

    def loss(self, X, y):
        dif = y - self.activation(X)
        return np.sum(dif**2/(2*len(y)), keepdims=True)

    def accuracy(self, X, y):
        pre = self.predict(X)
        return np.sum(np.where(pre==y, 1, 0))/len(y)

    def predict(self, X):
        result = np.where(self.activation(X)<0.5, 0.0, 1.0)
        return result

    # 解析的微分
    def update(self, X, y):
        a = (self.activation(X) - y)*self.activation(X)*(1 - self.activation(X))
        a = a.reshape(-1, 1)
        self.w -= self.eta * 1/float(len(y))*np.sum(a*X,axis=0)
        self.b -= self.eta * 1/float(len(y))*np.sum(a)

    # 数値微分
    def update_2(self, X, y):
        h = 1e-4
        loss_origin = self.loss(X, y)
        delta_w = np.zeros_like(self.w)

        for i in range(4):
            tmp = self.w[i]
            self.w[i] += h # パラメーターのうちの１つの値だけ少しだけ増加させる。
            loss_after = self.loss(X, y)
            delta_w[i] = self.eta*(loss_after - loss_origin)/h
            self.w[i] = tmp

        self.b += h
        loss_after = self.loss(X, y)
        delta_b = self.eta*(loss_after - loss_origin)/h
        self.w -= delta_w # 値の更新
        self.b -= delta_b

def plot(x, y, title, x_label, y_label, label, file_name):

    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for i, y_i in enumerate(y):
        plt.plot(x, y_i, label=label[i])

    plt.grid()
    plt.legend()

    print("Saving ==>> {}".format(file_name))
    plt.savefig(file_name)


def load_data():

    data = iris.data
    label = iris.target

    x1 = data[:40]
    x2 = data[50:90]
    x_train = np.concatenate((x1, x2), axis=0)

    ltr1 = label[:40]
    ltr2 = label[50:90]
    l_train = np.concatenate((ltr1, ltr2), axis=0)

    y1 = data[40:50]
    y2 = data[90:100]
    y_test = np.concatenate((y1, y2), axis=0)

    lte1 = label[40:50]
    lte2 = label[90:100]
    l_test = np.concatenate((lte1, lte2), axis=0)

    return x_train, l_train, y_test, l_test

def get_df_and_csv(result, columns, file_name):

    df = pd.DataFrame(result, columns=columns)

    print("Saving ==>> {}".format(file_name))
    df.to_csv(file_name, index=False)

    return df

def main():

    nn_1 = NeuralNetwork() #解析的微分
    nn_2 = NeuralNetwork() #数値微分

    epochs = 15

    X_train, y_train, X_test, y_test = load_data()

    result = []
    columns = ["epoch", "acc_1", "loss_1", "acc_2", "loss_2"]
    for epoch in range(epochs): # とりあえず15回ほど学習させてみる
        nn_1.update(X_train, y_train)
        nn_2.update_2(X_train, y_train)
        acc_1 = nn_1.accuracy(X_test, y_test)
        acc_2 = nn_2.accuracy(X_test, y_test)
        loss_1 = nn_1.loss(X_test, y_test)
        loss_2 = nn_2.loss(X_test, y_test)
        print('epoch  %d, acc_1  %.4f, loss_1 %.4f, acc_2 %.4f, loss_2 %.4f' % (epoch+1, acc_1, loss_1, acc_2, loss_2))
        result.append([epoch+1, acc_1, loss_1[0], acc_2, loss_2[0]])

    file_name = os.path.join(SAVE_RESULT_PATH, "nn_result.csv")
    df = get_df_and_csv(result, columns, file_name)

    plot(df["epoch"], [df["acc_1"], df["acc_2"]], "Accuracy", "epoch", "accuracy", ["analytic", "numerical"], os.path.join(SAVE_RESULT_PATH, "nn_accuracy.png"))
    plot(df["epoch"], [df["loss_1"], df["loss_2"]], "Loss", "epoch", "loss", ["analytic", "numerical"], os.path.join(SAVE_RESULT_PATH, "nn_loss.png"))

    weights_1 = nn_1.get_weight()
    weights_2 = nn_2.get_weight()
    bias_1 = nn_1.get_bias()
    bias_2 = nn_2.get_bias()
    print('weights_1 = ', weights_1, 'bias_1 = ', bias_1)
    print('weights_2 = ', weights_2, 'bias_2 = ', bias_2)

    file_name = os.path.join(SAVE_RESULT_PATH, "nn_weight_bias.txt")
    print("Saving ==>> {}".format(file_name))
    with open(file_name, mode="w") as f:
        f.write("weights_1: {}, bias_1: {}\n".format(weights_1, bias_1))
        f.write("weights_2: {}, bias_2: {}\n".format(weights_2, bias_2))

if __name__ == "__main__":
    main()