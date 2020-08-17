################# Aluno: João Vitor Roma Neto #############
################# Matrícula: 201600560171 #################
###########################################################
import numpy as np
import pandas as pd
import constants as C
import matplotlib.pyplot as plt


def train_data(file_name: str) -> bool:
    """
    Treina o dataset dado por filename
    - entrada: x = {x_1, x_2, ..., x_n}
    - parâmetros: t = {w_1, w_2, ... w_n, b}
    - ativação: sigma(f(x))
    """
    df = pd.read_csv(file_name, names=["x1", "x2", "y"])

    X = add_bias(get_X(df)).to_numpy(dtype=np.double)  # obtém X e adiciona o bias
    Y = get_Y(df).to_numpy()
    w = generate_weight(X)
    m = len(X)
    for epoch in range(C.EPOCH):
        y = sigmoid(np.dot(X, w))
        for i in range(m):
            fixed = (1 / m) * C.LAMBDA * ((2 * (y[i] - Y[i])) * (y[i] - y[i] ** 2))
            w[0] = w[0] - fixed * X[i, 0]
            w[1] = w[1] - fixed * X[i, 1]
            w[2] - w[2] - fixed

    plot_all(X, Y, w)

    print("{0}%".format(predict(y, Y, m) * 100))


def predict(y, Y, m):
    """
    Compara os valores de y com Y e calcula os acertos
    """
    y = list(map(lambda x: 1 if x > 0.5 else -1, y))
    return len(np.where(y == Y)[0]) / m


def generate_weight(X):
    """
    Gera peso aleatório no intervalo [-1, 1]
    """
    return np.random.uniform(C.WEIGHT_MIN, C.WEIGHT_MAX, X.shape[1])


def get_Y(df):
    """
    Obtém a útlima coluna do csv
    """
    return df.iloc[:, -1]


def get_X(df):
    """
    Obtém as duas primeiras colunas do csv
    """
    return df.iloc[:, :-1]


def add_bias(X):
    """
    adiciona uma coluna de 1 em X
    """
    r, c = X.shape  # row, column
    bias = np.ones(r)
    X["bias"] = bias
    return X


def sigmoid(x):
    """
    Aplica a função sigmoid para X
    """
    return 1 / (1 + np.exp(-x))


def plot_all(X, Y, w):
    pos_X = np.take(X[:, 0], np.where(Y == 1))
    pos_Y = np.take(X[:, 1], np.where(Y == 1))
    neg_X = np.take(X[:, 0], np.where(Y == -1))
    neg_Y = np.take(X[:, 1], np.where(Y == -1))
    plt.plot(pos_X, pos_Y, "+r")
    plt.plot(neg_X, neg_Y, "ob")

    xx = np.linspace(-3, 4)  # hyperplane? '-''
    plt.plot(xx, (w[0] * xx + w[1]))  # dúvida
    plt.show()

