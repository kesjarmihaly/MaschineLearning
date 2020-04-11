import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from urllib import request
import os
import gzip

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'x_train':'train-images-idx3-ubyte.gz',
    't_train':'train-labels-idx1-ubyte.gz',
    'x_test':'t10k-images-idx3-ubyte.gz',
    't_test':'t10k-labels-idx1-ubyte.gz'
}

DATA_PATH = './data/'

if not os.path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH)

    for v in key_file.values():
        file_path = DATA_PATH + v
        print("Loading >> {}".format(file_path))
        request.urlretrieve(url_base + v, file_path)

SAVE_RESULT_PATH = "./result/"
if not os.path.isdir(SAVE_RESULT_PATH):
    os.makedirs(SAVE_RESULT_PATH)


def load_label(file_name):
    file_path = file_name
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
            # 最初の８バイト分はデータ本体ではないので飛ばす
    one_hot_labels = np.zeros((labels.shape[0], 10))
    for i in range(labels.shape[0]):
        one_hot_labels[i, labels[i]] = 1
    return one_hot_labels

def load_image(file_name):
    file_path = file_name
    with gzip.open(file_path, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
        # 画像本体の方は16バイト分飛ばす必要がある
    return images

def convert_into_numpy(key_file):
    dataset = {}

    dataset['x_train'] = load_image(os.path.join(DATA_PATH, key_file['x_train']))
    dataset['t_train'] = load_label(os.path.join(DATA_PATH, key_file['t_train']))
    dataset['x_test']  = load_image(os.path.join(DATA_PATH, key_file['x_test']))
    dataset['t_test']  = load_label(os.path.join(DATA_PATH, key_file['t_test']))

    return dataset

def load_mnist():
    # mnistを読み込みNumPy配列として出力する
    dataset = convert_into_numpy(key_file)
    dataset['x_train'] = dataset['x_train'].astype(np.float32) # データ型を`float32`型に指定しておく
    dataset['x_test'] = dataset['x_test'].astype(np.float32)
    dataset['x_train'] /= 255.0
    dataset['x_test'] /= 255.0 # 簡単な標準化
    dataset['x_train'] = dataset['x_train'].reshape(-1, 28*28)
    dataset['x_test']  = dataset['x_test'].reshape(-1, 28*28)
    return dataset


def sigmoid(x):  # シグモイド関数
    return 1 / (1 + np.exp(-x))

def inner_product(X, w, b):  # ここは内積とバイアスを足し合わせる
    return np.dot(X, w) + b

def activation(X, w, b):
    return sigmoid(inner_product(X, w, b))


class DeepNeuralNetwork:

    def __init__(self, shape_list=[784, 100, 10], eta=2.0):

        self.weight_list, self.bias_list = self.make_params(shape_list)
        self.eta = eta

    def make_params(self, shape_list): # shape_list = [784, 100, 10]のように層ごとにニューロンの数を配列にしたものを入力する
        weight_list = []
        bias_list = []
        for i in range(len(shape_list)-1):
            weight = np.random.randn(shape_list[i], shape_list[i+1]) # 標準正規分布に従った乱数を初期値とする
            bias = np.ones(shape_list[i+1])/10.0 # 初期値はすべて0.1にする
            weight_list.append(weight)
            bias_list.append(bias)
        return weight_list, bias_list

    def calculate(self, X, t):
        val_list = {}
        a_1 = inner_product(X, self.weight_list[0], self.bias_list[0]) # (N, 1000)
        y_1 = sigmoid(a_1)
        a_2 = inner_product(y_1, self.weight_list[1], self.bias_list[1]) # (N, 10)
        y_2 = sigmoid(a_2)
        y_2 /= np.sum(y_2, axis=1, keepdims=True) # ここで簡単な正規化をはさむ
        S = 1/(2*len(y_2))*(y_2 - t)**2
        L = np.sum(S)
        val_list['a_1'] = a_1
        val_list['y_1'] = y_1
        val_list['a_2'] = a_2
        val_list['y_2'] = y_2
        val_list['S'] = S
        val_list['L'] = L

        return val_list

    def predict(self, X, t):
        val_list = self.calculate(X, t)
        y_2 = val_list['y_2']
        result = np.zeros_like(y_2)
        for i in range(y_2.shape[0]): # サンプル数にあたる
            result[i, np.argmax(y_2[i])] = 1
        return result

    def accuracy(self, X, t):
        pre = self.predict(X, t)
        result = np.where(np.argmax(t, axis=1)==np.argmax(pre, axis=1), 1, 0)
        acc = np.mean(result)
        return acc

    def loss(self, X, t):
        L = self.calculate(X, t)['L']
        return L

    def update(self, X, t): # etaは学習率。ここでパラメータの更新を行う

        val_list = self.calculate(X, t)
        a_1 = val_list['a_1']
        y_1 = val_list['y_1']
        a_2 = val_list['a_2']
        y_2 = val_list['y_2']
        S = val_list['S']
        L = val_list['L']

        dL_dS = 1.0
        dS_dy_2 = 1/X.shape[0]*(y_2 - t)
        dy_2_da_2 = y_2*(1.0 - y_2)
        da_2_dw_2 = np.transpose(y_1)
        da_2_db_2 = 1.0
        da_2_dy_1 = np.transpose(self.weight_list[1])
        dy_1_da_1 = y_1 * (1 - y_1)
        da_1_dw_1 = np.transpose(X)
        da_1_db_1 = 1.0

        # ここからパラメータの更新を行っていく。
        dL_da_2 =  dL_dS * dS_dy_2 * dy_2_da_2
        self.bias_list[1] -= self.eta*np.sum(dL_da_2 * da_2_db_2, axis=0)
        self.weight_list[1] -= self.eta*np.dot(da_2_dw_2, dL_da_2)
        dL_dy_1 = np.dot(dL_da_2, da_2_dy_1)
        dL_da_1 = dL_dy_1 * dy_1_da_1
        self.bias_list[0] -= self.eta*np.sum(dL_da_1 * da_1_db_1, axis=0)
        self.weight_list[0] -= self.eta*np.dot(da_1_dw_1, dL_da_1)

def get_df_and_csv(result, columns, file_name):

    df = pd.DataFrame(result, columns=columns)

    print("Saving ==>> {}".format(file_name))
    df.to_csv(file_name, index=False)

    return df

def visualize_data(dataset, file_name):
    plt.figure(figsize=(10, 8))

    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.imshow(dataset['x_train'][i, :].reshape(28, 28))
        plt.subplots_adjust(wspace=0.4, hspace=0.6)

    print("Saving ==>> {}".format(file_name))
    plt.savefig(file_name)

def plot(x, y, title, x_label, y_label, file_name):

    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.plot(x, y)

    plt.grid()

    print("Saving ==>> {}".format(file_name))
    plt.savefig(file_name)


def main():

    epochs = 100
    batch_size = 1000
    total_acc_list = []
    total_loss_list = []

    dnn = DeepNeuralNetwork()

    dataset = load_mnist()

    file_name = os.path.join(SAVE_RESULT_PATH, "mnist.png")
    visualize_data(dataset, file_name)

    X_train, t_train, X_test, t_test = dataset["x_train"], dataset["t_train"], dataset["x_test"], dataset["t_test"]

    columns = ["Epoch", "Accuracy", "Loss"]

    results = []
    for epoch in range(epochs):

        for i in range(60000//batch_size):
            ra = np.random.randint(60000, size=batch_size) # 0~59999でランダムな整数をbatch_size分だけ発生させる。
            x_batch, t_batch = X_train[ra,:], t_train[ra,:]
            dnn.update(x_batch, t_batch)

        acc_list = []
        loss_list = []
        for k in range(10000//batch_size):
            x_batch, t_batch = X_test[k*batch_size:(k+1)*batch_size, :], t_test[k*batch_size:(k+1)*batch_size, :]
            acc_val = dnn.accuracy(x_batch, t_batch)
            loss_val = dnn.loss(x_batch, t_batch)
            acc_list.append(acc_val)
            loss_list.append(loss_val)
        acc = np.mean(acc_list)   # 精度は平均で求める
        loss = np.mean(loss_list) # 損失は合計で求める。
        total_acc_list.append(acc)
        total_loss_list.append(loss)

        print("Epoch: %d, Accuracy: %f, Loss: %f" % (epoch+1, acc, loss))
        results.append([epoch+1, acc, loss])

    file_name = os.path.join(SAVE_RESULT_PATH, "dnn_result.csv")
    df = get_df_and_csv(results, columns, file_name)

    file_name = os.path.join(SAVE_RESULT_PATH, "dnn_accuracy.png")
    plot(df["Epoch"], df["Accuracy"], "Accuracy", "epoch", "accuracy", file_name)

    file_name = os.path.join(SAVE_RESULT_PATH, "dnn_loss.png")
    plot(df["Epoch"], df["Loss"], "Loss", "epoch", "loss", file_name)


if __name__ == "__main__":
    main()