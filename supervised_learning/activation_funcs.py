import numpy as np

def softmax(z):
    exp_z = np.exp(z)               #element-wise exponential (z is vector of linear transformations of output layer)
    soft_m = exp_z/np.sum(exp_z)    #element-wise np.exp(zi)/sum(e^zi...zn)
    return soft_m                   #return vector containing e^zi/sum(e^zi...zn)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(Z):
    g_z = np.maximum(0, Z)
    return g_z