import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA

def _images(path):
    """Return images loaded locally."""
    with gzip.open(path) as f:
        # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
        pixels = np.frombuffer(f.read(), 'B', offset=16)
    return pixels.reshape(-1, 784).astype('float32') / 255

def _labels(path, onehot=False):
    """Return labels loaded locally."""
    with gzip.open(path) as f:
        # First 8 bytes are magic_number, n_labels
        integer_labels = np.frombuffer(f.read(), 'B', offset=8)

    def _onehot(integer_labels):
        """Return matrix whose rows are onehot encodings of integers."""
        n_rows = len(integer_labels)
        n_cols = integer_labels.max() + 1
        onehot = np.zeros((n_rows, n_cols), dtype='uint8')
        onehot[np.arange(n_rows), integer_labels] = 1
        return onehot

    if onehot:
        return _onehot(integer_labels)
    else:
        return integer_labels

class RandomModel():
    def __init__(self):
        self.mean = None
        self.std = None
        # This sets up the neural net. It's by far the most fun part to play with
        self.net = nn.Sequential(
            # starting layer
            nn.Linear(784,392),
            nn.ReLU(),

            # layer 2
            nn.BatchNorm1d(392),
            nn.Linear(392, 392),
            nn.ReLU(),

            # layer 3
            nn.BatchNorm1d(392),
            nn.Linear(392, 392),
            nn.ReLU(),

            # layer 4
            nn.BatchNorm1d(392),
            nn.Linear(392, 392),
            nn.ReLU(),

            # layer 5
            nn.BatchNorm1d(392),
            nn.Linear(392, 392),
            nn.ReLU(),

            # layer 6
            nn.BatchNorm1d(392),
            nn.Linear(392, 392),
            nn.ReLU(),

            # layer 7
            nn.BatchNorm1d(392),
            nn.Linear(392, 392),
            nn.ReLU(),

            # layer 8
            nn.BatchNorm1d(392),
            nn.Linear(392, 392),
            nn.ReLU(),

            # layer 9
            nn.BatchNorm1d(392),
            nn.Linear(392, 392),
            nn.ReLU(),

            # layer 10
            nn.BatchNorm1d(392),
            nn.Linear(392,10)
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), 1e-4)


    def preprocess(self, X):
        X = (X - self.mean) / self.std
        # X = PCA(256).fit_transform(X)
        return X

    def train(self, X, Y):
        self.mean = X.mean(0)
        self.std = X.std(0)
        Xpr = self.preprocess(X)
        Xt = torch.tensor(Xpr).float()
        Yt = torch.tensor(Y)
        #Yt = Yt.reshape(Yt.shape[0], 1)
        lossf = nn.CrossEntropyLoss()

        for i in range (400):
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            Yhat = self.net(Xt)
            loss = lossf(Yhat,Yt)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            print(loss, i)

    def predict(self, X):
        Xt = torch.tensor(X).float()
        outputs = self.net(Xt)
        _, predictions = torch.max(outputs, 1)
        return predictions
    
    def evaluate(self, X, Y):
        Xpr = self.preprocess(X)
        Yhat = self.predict(Xpr)
        correct = (Yhat == Y)
        return 100*correct.numpy().astype('float').mean()
    

if __name__ == '__main__':
    global model

    model = RandomModel()
    mnist_trainX_np = _images("train-images-idx3-ubyte.gz")
    mnist_trainY_np = _labels("train-labels-idx1-ubyte.gz", onehot=False)

    model.train(mnist_trainX_np, mnist_trainY_np)
    print(model.predict(mnist_trainX_np))
    print(model.evaluate(mnist_trainX_np, mnist_trainY_np))


    # trainX1,trainX2,valX = np.split(mnist_trainX_np,3)
    # trainY1,trainY2,valY = np.split(mnist_trainY_np,3)


    # trainX = np.concatenate((trainX1,trainX2))
    # trainY = np.concatenate((trainY1,trainY2))

    # valX1,valX2,valX3,valX4 = np.split(valX,4)
    # valY1,valY2,valY3,valY4 = np.split(valY,4)

    # valX123 = np.concatenate((valX1,valX2,valX3))
    # valX234 = np.concatenate((valX2,valX3,valX4))
    # valX134 = np.concatenate((valX1,valX3,valX4))
    # valX124 = np.concatenate((valX1,valX2,valX4))

    # valY123 = np.concatenate((valY1,valY2,valY3))
    # valY234 = np.concatenate((valY2,valY3,valY4))
    # valY134 = np.concatenate((valY1,valY3,valY4))
    # valY124 = np.concatenate((valY1,valY2,valY4))

    # model.train(mnist_trainX_np, mnist_trainY_np)
    # # print(model.evaluate(mnist_trainX_np,mnist_trainY_np))

    # # predictions = model.predict(mnist_trainX_np)
    # # print("Predictions:", predictions)

    # # model.train(trainX, trainY, epochs=10, print_every=1)

    # print(model.evaluate(valX123,valY123))
    # print(model.evaluate(valX234,valY234))
    # print(model.evaluate(valX134,valY134))
    # print(model.evaluate(valX124,valY124))