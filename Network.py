import numpy as np

#the sigmoid function 1/(1+e^(-z))
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class Network(object):

#builder of a network - lyrs is how many nodes in each lyr-
    def __init__(self, lyrs):
        self.num_lyrs = len(lyrs)
        self.lyrs = lyrs
        self.baises = [np.random.randn(y, 1) for y in lyrs[1:]] #for each lyr of nodes build an array of baises
        self.weights = [np.random.randn(y, x) for x,y in zip(lyrs[:-1],lyrs[1:])]#builds weight lists between the lyrs/

    def feedforword(self,a):
        for b,w in zip(self.baises, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

