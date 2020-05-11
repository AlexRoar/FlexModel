import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn import datasets
import sklearn
from tqdm.notebook import tqdm

class Dense:
    compute_type = 'layer'

    def sigmoid(self, Z):
        return np.divide(1, 1 + np.exp(-Z))

    def sigmoid_backward(self, Z):
        return self.sigmoid(Z) * (1 - self.sigmoid(Z))

    def relu(self, Z):
        return np.maximum(Z, 0)

    def relu_backward(self, Z):
        dZ = np.array(Z, copy=True)
        dZ[Z <= 0] = 0
        dZ[Z > 0] = 1
        return dZ

    def lerelu(self, Z):
        return np.where(Z > 0, Z, Z * 0.001)

    def lerelu_backward(self, Z):
        return np.where(Z > 0, 1, 0.001)

    def tanh(self, Z):
        return np.tanh(Z)

    def tanh_backward(self, Z):
        return 1 - np.power(self.tanh(Z), 2)

    def softmax(self, Z):
        A = np.exp(Z)
        A /= np.sum(A)
        return A

    def softmax_backward(self, Z):
        return self.sigmoid_backward(Z)

    def init_params(self, prev_layer):
        if self.activation != 'relu' and self.activation != 'lerelu':
            self.w = np.random.randn(self.neurons, int(prev_layer)) * np.sqrt(1 / int(prev_layer))*0.01
        else:
            self.w = np.random.randn(self.neurons, int(prev_layer)) * np.sqrt(2 / int(prev_layer))*0.01
        self.w = self.w.astype(np.longlong)
        self.b = np.zeros((self.neurons, 1)).astype(np.longlong)

    def set_func_alias(self):
        if self.activation == 'tanh':
            self.forwardFunc = self.tanh
            self.backFunc = self.tanh_backward
        elif self.activation == 'sigmoid':
            self.forwardFunc = self.sigmoid
            self.backFunc = self.sigmoid_backward
        elif self.activation == 'relu':
            self.forwardFunc = self.relu
            self.backFunc = self.relu_backward
        elif self.activation == 'lerelu':
            self.forwardFunc = self.lerelu
            self.backFunc = self.lerelu_backward
        elif self.activation == 'softmax':
            self.forwardFunc = self.softmax
            self.backFunc = self.softmax_backward

    def perform_dropout(self, A):
        if self.dropout >= 1:
            return A
        Dl = np.random.rand(*A.shape)

        keep_prob = float(self.dropout)
        Dl = (Dl < keep_prob).astype(int)

        Al = A * Dl
        Al /= keep_prob

        self.cache['D'] = Dl
        return Al

    def perform_dropout_back(self, dA):
        if self.dropout >= 1:
            return dA
        keep_prob = float(self.dropout)

        Dl = self.cache['D']
        dA *= Dl
        dA /= keep_prob

        return dA

    def forward_propagation(self, A_prev):
        A_prev = A_prev.copy().astype(np.longlong)
        self.cache['A_prev'] = A_prev

        Z = self.w.dot(A_prev) + self.b
        A = self.forwardFunc(Z)

        A = self.perform_dropout(A)

        self.cache['Z'] = Z
        self.cache['A'] = A
        return A.copy()

    def backpropagation(self, dA, m, learning_rate, optimization, data):
        dA = dA.copy().astype(np.longlong)
        dA = self.perform_dropout_back(dA)
        dZ = dA * self.backFunc(self.cache['Z'])

        dW = 1 / m * np.dot(dZ, self.cache['A_prev'].T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        assert (self.w.shape == dW.shape)
        assert (self.b.shape == db.shape)

        dW += self.lambd / m * self.w

        if optimization == None:
            pass
        elif optimization == 'momentum':
            beta = data['beta']
            t = data['epoch'] + 1

            self.vdw = beta * self.vdw + (1 - beta) * dW
            self.vdb = beta * self.vdb + (1 - beta) * db

            dW = self.vdw / (1 - beta ** t)
            db = self.vdb / (1 - beta ** t)
        elif optimization == 'rmsprop':
            beta = data['beta']
            t = data['epoch'] + 1

            self.sdw = beta * self.sdw + (1 - beta) * dW * dW
            self.sdb = beta * self.sdb + (1 - beta) * db * db

            dW = dW / np.sqrt(self.sdw + self.epsilon) / (1 - beta ** t)
            db = db / np.sqrt(self.sdw + self.epsilon) / (1 - beta ** t)
        elif optimization == 'adam':
            beta1 = data['beta1']
            beta2 = data['beta2']
            t = data['epoch'] + 1

            self.vdw = beta1 * self.vdw + (1 - beta1) * dW
            self.vdb = beta1 * self.vdb + (1 - beta1) * db
            self.sdw = beta2 * self.sdw + (1 - beta2) * dW * dW
            self.sdb = beta2 * self.sdb + (1 - beta2) * db * db

            vdw_corr = self.vdw / (1 - beta1 ** t)
            vdb_corr = self.vdb / (1 - beta1 ** t)
            sdw_corr = self.sdw / (1 - beta2 ** t)
            sdb_corr = self.sdb / (1 - beta2 ** t)

            dW = vdw_corr / (np.sqrt(sdw_corr) + self.epsilon)
            db = vdb_corr / (np.sqrt(sdb_corr) + self.epsilon)

        dW += self.lambd / m * self.w

        dA_prev = np.dot(self.w.T, dZ)
        self.w -= learning_rate * dW
        self.b -= self.b - learning_rate * db
        return dA_prev.copy().astype(np.longlong)

    def __init__(self, n_neurons=1, activation='tanh', dropout=1, prev_layer=None, batchnorm=False, epsilon=1e-10):
        self.activation = str(activation).lower()
        self.neurons = int(n_neurons)
        self.dropout = float(dropout)
        self.batchnorm = batchnorm
        self.epsilon = epsilon
        self.lambd = 0
        self.w = np.array([])
        self.b = np.array([])

        self.cache = {}

        self.vdw = 0
        self.vdb = 0

        self.sdw = 0
        self.sdb = 0

        self.forwardFunc = None
        self.backFunc = None

        if not prev_layer is None:
            self.init_params(prev_layer)
        self.set_func_alias()


class FlexModel:

    def __init__(self):
        self.layers = []
        self.preprocess = []
        self.lossF = self.cross_entropy_loss
        self.lossFBack = lossF = self.d_cross_entropy_loss

    def preprocessData(self, X, y):
        for ob in self.preprocess:
            if not X is None:
                X = ob.processX(X)
            if not y is None:
                y = ob.processY(y)
        return X, y

    def add(self, layer):
        if layer.compute_type == 'layer':
            self.layers.append(layer)
        else:
            self.preprocess.append(layer)

    def splitBatchesNum(self, X, num):
        size = X.shape[1] // num
        return self.splitBatchesSize(X, size)

    def splitBatchesSize(self, X, size):
        batches = []
        if size >= X.shape[1]:
            return [X]
        for i in range(X.shape[1] // size):
            batches.append(X[:, size * i: size * (i + 1)])
        if X.shape[1] % size != 0:
            batches.append(X[:, size * (i + 1):])
        return batches

    def cross_entropy_loss(self, A, Y, m):
        m = Y.shape[1]
        cost = np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) * (-1 / m)
        cost = np.squeeze(cost)
        return cost

    def d_cross_entropy_loss(self, A, Y):
        return -np.divide(Y, A) + np.divide(1 - Y, 1 - A)

    def forward_propagation(self, A):
        A = A.copy()
        for n, layer in enumerate(self.layers):
            A = self.layers[n].forward_propagation(A)
        return A

    def backpropagation(self, dAL, learning_rate, m, optimization, data):
        dAl = dAL.copy()
        L = len(self.layers)
        for n, layer in enumerate(self.layers):
            dAl = self.layers[L - n - 1].backpropagation(dAl, m, learning_rate, optimization, data)

    def fit(self, X, y, learning_rate=0.01, batches_size=None, lambd=0, n_iter=1500, printLoss=True, decay_rate=0,
            optimization='adam', beta1=0.9, beta2=0.999, printEvery=10):
        X, y = self.preprocessData(X, y)
        X = np.array(X).astype(np.longlong)
        y = np.array(y).astype(np.longlong)
        optimization = str(optimization).lower()
        learning_rate = float(learning_rate)
        lambd = float(lambd)
        n_iter = int(n_iter)
        decay_rate = float(decay_rate)
        beta1 = float(beta1)
        beta2 = float(beta2)
        data = {'beta': beta1, 'beta1': beta1, 'beta2': beta2}

        if batches_size is None:
            batchesX = [X]
            batchesY = [y]
        else:
            batches_size = int(batches_size)
            batchesX = self.splitBatchesSize(X, batches_size)
            batchesY = self.splitBatchesSize(y, batches_size)

        prevSize = X.shape[0]
        for n, layer in enumerate(self.layers):
            self.layers[n].init_params(prevSize)
            self.layers[n].lambd = lambd
            prevSize = layer.neurons

        progress = []
        for epoch in tqdm(range(n_iter)):
            learning_rate_degraded = 1/(1+decay_rate * epoch) * learning_rate
            data['epoch'] = epoch
            loses = []
            for b in range(len(batchesX)):
                batchX = batchesX[b]
                batchY = batchesY[b]
                m = batchX.shape[1]

                AL = self.forward_propagation(batchX)
                loss = self.lossF(AL, batchY, m)
                loses.append(loss)
                progress.append(loss)
                dAL = self.lossFBack(AL, batchY)
                self.backpropagation(dAL, learning_rate=learning_rate_degraded, m=m, optimization=optimization,
                                     data=data)
            if printLoss and epoch % printEvery == 0:
                out = 'Epoch %s: %s' % (epoch, np.mean(loses))
                if decay_rate != 0:
                    out += ' | learning rate: %s' % (round(learning_rate_degraded, 5))
                print(out)
        return progress

    def predict(self, X):
        X, _ = self.preprocessData(X, None)
        X = np.array(X)
        return self.forward_propagation(X)

    def accuracy(self, A, Y):
        corr = 0
        for i in range(A.shape[1]):
            if A[:, i] == Y[:, i]:
                corr += 1
        return corr / A.shape[1]


class Preprocessor:
    compute_type = 'preprocess'

    def __init__(self, funcX=None, funcY=None):
        self.funcX = funcX
        self.funcY = funcY

    def processX(self, X):
        if self.funcX is None:
            return X
        return self.funcX(X)

    def processY(self, y):
        if self.funcY is None:
            return y
        return self.funcY(y)
