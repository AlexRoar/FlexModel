import numpy as np
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

    def linear(self, Z):
        return Z

    def linear_backward(self, Z):
        return 1

    def softmax(self, Z):
        A = np.exp(Z)
        A /= np.sum(A, axis=0, keepdims=True)
        return A

    def softmax_backward(self, Z):
        return self.sigmoid_backward(Z)

    def init_params(self, prev_layer):
        if self.activation != 'relu' and self.activation != 'lerelu':
            self.w = np.random.randn(self.neurons, int(prev_layer)) * np.sqrt(1 / int(prev_layer)) * 0.01
        else:
            self.w = np.random.randn(self.neurons, int(prev_layer)) * np.sqrt(2 / int(prev_layer)) * 0.01
        self.w = self.w.astype(np.longdouble)
        self.b = np.zeros((self.neurons, 1)).astype(np.longdouble)

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
        elif self.activation == 'linear':
            self.forwardFunc = self.linear
            self.backFunc = self.linear_backward


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

    def forward_propagation(self, A_prev, final=False):
        A_prev = A_prev.copy().astype(np.longdouble)
        self.cache['A_prev'] = A_prev

        Z = self.w.dot(A_prev) + self.b
        A = self.forwardFunc(Z)

        if not final:
            A = self.perform_dropout(A)

        self.cache['Z'] = Z
        self.cache['A'] = A
        return A.copy()

    def backpropagation(self, dA, m, learning_rate, optimization, data):
        dA = dA.copy().astype(np.longdouble)
        dA = self.perform_dropout_back(dA)
        dZ = dA * self.backFunc(self.cache['Z'])

        dW = 1 / m * np.dot(dZ, self.cache['A_prev'].T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        assert (self.w.shape == dW.shape)
        assert (self.b.shape == db.shape)

        dW += self.lambd / m * self.w
        t = data['epoch'] + 1
        if optimization == None:
            pass
        elif optimization == 'momentum':
            beta = data['beta']

            self.vdw = beta * self.vdw + (1 - beta) * dW
            self.vdb = beta * self.vdb + (1 - beta) * db

            dW = self.vdw / (1 - beta ** t)
            db = self.vdb / (1 - beta ** t)
        elif optimization == 'rmsprop':
            beta = data['beta']

            self.sdw = beta * self.sdw + (1 - beta) * dW * dW
            self.sdb = beta * self.sdb + (1 - beta) * db * db

            dW = dW / np.sqrt(self.sdw + self.epsilon) / (1 - beta ** t)
            db = db / np.sqrt(self.sdw + self.epsilon) / (1 - beta ** t)
        elif optimization == 'adam':
            beta1 = data['beta1']
            beta2 = data['beta2']

            self.vdw = np.longdouble(beta1 * self.vdw + (1 - beta1) * dW)
            self.vdb = np.longdouble(beta1 * self.vdb + (1 - beta1) * db)
            self.sdw = np.longdouble(beta2 * self.sdw + (1 - beta2) * dW * dW)
            self.sdb = np.longdouble(beta2 * self.sdb + (1 - beta2) * db * db)

            vdw_corr = np.longdouble(self.vdw / (1 - beta1 ** t))
            vdb_corr = np.longdouble(self.vdb / (1 - beta1 ** t))
            sdw_corr = np.longdouble(self.sdw / (1 - beta2 ** t))
            sdb_corr = np.longdouble(self.sdb / (1 - beta2 ** t))

            dW = np.longdouble(vdw_corr) / np.longdouble(np.sqrt(sdw_corr) + self.epsilon)
            db = np.longdouble(vdb_corr) / np.longdouble(np.sqrt(sdb_corr) + self.epsilon)

        dW += self.lambd / m * self.w

        dA_prev = np.dot(self.w.T, dZ)
        self.w = np.subtract(np.longdouble(self.w), np.longdouble(learning_rate * dW))
        self.b = np.subtract(np.longdouble(self.b), np.longdouble(learning_rate * db))

        return dA_prev.copy().astype(np.longdouble)

    def __init__(self, n_neurons=1, activation='tanh', dropout=1, prev_layer=None, batchnorm=False, epsilon=1e-19):
        self.activation = str(activation).lower()
        self.neurons = int(n_neurons)
        self.dropout = np.longdouble(dropout)
        self.batchnorm = batchnorm
        self.epsilon = np.longdouble(epsilon)
        self.lambd = 0
        self.w = np.array([]).astype(np.longdouble)
        self.b = np.array([]).astype(np.longdouble)

        self.cache = {}

        self.vdw = np.longdouble(0)
        self.vdb = np.longdouble(0)

        self.sdw = np.longdouble(0)
        self.sdb = np.longdouble(0)

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

    @staticmethod
    def normalize_outbreaks(A, min_val=0, max_val=1, eps=1e-15):
        A[A <= min_val] = min_val + eps
        A[A >= max_val] = max_val - eps
        return A

    def cross_entropy_loss(self, A, Y, eps=1e-15):
        m = Y.shape[1]
        if len(A[A <= 0]) != 0 or len(A[A >= 1]) != 0:
            A = self.normalize_outbreaks(A, eps=eps)
        cost = np.sum(Y * np.log(A + eps) + (1 - Y) * np.log(1 - A + eps)) * (-1 / m)
        cost = np.squeeze(cost)
        return cost

    def d_cross_entropy_loss(self, A, Y, eps=1e-15):
        if len(A[A <= 0]) != 0 or len(A[A >= 1]) != 0:
            A = self.normalize_outbreaks(A, eps=eps)
        return -np.divide(Y, A) + np.divide(1 - Y, 1 - A)

    def forward_propagation(self, A, final=False):
        A = A.copy()
        for n, layer in enumerate(self.layers):
            A = self.layers[n].forward_propagation(A, final=final)
        return A

    def backpropagation(self, dAL, learning_rate, m, optimization, data):
        dAl = dAL.copy()
        L = len(self.layers)
        for n, layer in enumerate(self.layers):
            dAl = self.layers[L - n - 1].backpropagation(dAl, m, learning_rate, optimization, data)

    def decayFunc(self, epoch):
        if self.decay_type == 'exponential':
            return self.decay_rate ** epoch * self.learning_rate
        elif self.decay_type == 'hyperbolic':
            return 1 / (1 + self.decay_rate * epoch) * self.learning_rate
        elif self.decay_type == 'squared':
            return self.decay_rate / np.sqrt(epoch) * self.learning_rate
        elif self.decay_type == None:
            return self.learning_rate
        else:
            raise Exception('Decay type is not defined')

    def fit(self, X, y, learning_rate=0.01, batches_size=None, lambd=0, n_iter=1500, printLoss=True, decay_rate=0,
            optimization='adam', beta1=0.9, beta2=0.999, printEvery=10, decay_type='exponential', eval_set=None,
            eval_every=10):
        X, y = self.preprocessData(X, y)
        X = np.array(X).astype(np.longdouble)
        y = np.array(y).astype(np.longdouble)

        optimization = str(optimization).lower()

        learning_rate = np.longdouble(learning_rate)
        self.learning_rate = learning_rate

        lambd = np.longdouble(lambd)

        n_iter = int(n_iter)

        decay_rate = np.longdouble(decay_rate)
        self.decay_rate = decay_rate
        self.decay_type = str(decay_type)

        beta1 = np.longdouble(beta1)
        beta2 = np.longdouble(beta2)

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

        if eval_set is not None:
            eval_X, eval_Y = self.preprocessData(eval_set[0], eval_set[1])

        progress = [[],[]]
        progressEval = [[],[]]
        for epoch in tqdm(range(1, n_iter + 1)):
            learning_rate_degraded = self.decayFunc(epoch)
            data['epoch'] = epoch
            loses = []
            for b in range(len(batchesX)):
                batchX = batchesX[b]
                batchY = batchesY[b]
                m = batchX.shape[1]

                AL = self.forward_propagation(batchX)
                loss = self.lossF(AL, batchY)
                loses.append(loss)

                dAL = self.lossFBack(AL, batchY)
                self.backpropagation(dAL, learning_rate=learning_rate_degraded, m=m, optimization=optimization,
                                     data=data)
            progress[0].append(epoch)
            progress[1].append(np.mean(loses))

            if epoch % eval_every == 0 and eval_set is not None:
                eval_pred = self.forward_propagation(eval_X)
                lossEval = self.lossF(eval_pred, eval_Y)
                progressEval[0].append(epoch)
                progressEval[1].append(lossEval)
            if printLoss and epoch % printEvery == 0:
                out = 'Epoch %s: %s' % (epoch, np.mean(loses))
                if decay_rate != 0:
                    out += ' | learning rate: %s' % (round(learning_rate_degraded, 5))
                if eval_set is not None:
                    eval_pred = self.forward_propagation(eval_X)
                    lossEval = self.lossF(eval_pred, eval_Y)
                    out += ' | eval loss: %s' % (lossEval)
                print(out)
        if eval_set is None:
            return progress
        else:
            return progress, progressEval

    def predict(self, X):
        X, _ = self.preprocessData(X, None)
        X = np.array(X)
        return self.forward_propagation(X, final=True)

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
