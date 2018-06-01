def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

import matplotlib.pyplot as plt
from scipy.misc import imsave
import numpy as np

def cifar10Matrix(row):
    img = np.reshape(row, (3,32,32))
    img = img.transpose(1,2,0)
    return img

class cifar10(object):

    def __init__(self, mode='train'):
        self.row = -1
        self.X = []
        self.Y = []
        self.bound = 10000
        if(mode is 'train'):
            for i in range(1,6):
                data = unpickle("D:/dataset/cifar-10-python/cifar-10-batches-py/data_batch_"+str(i))
                for i in range(0, self.bound):
                    self.X.append(cifar10Matrix(data[b'data'][i]))
                    self.Y.append(data[b'labels'][i])
        elif(mode == 'test'):
            data = unpickle("D:/dataset/cifar-10-python/cifar-10-batches-py/test_batch")
            for i in range(0, self.bound):
                    self.X.append(cifar10Matrix(data[b'data'][i]))
                    self.Y.append(data[b'labels'][i])
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        
    def cifar10Input(self):
        self.row += 1
        return (self.X[self.row], self.Y[self.row])    
    def cifar10All(self):
        return (self.X, self.Y)
        