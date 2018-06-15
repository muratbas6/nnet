import numpy as np


def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))


X = np.array([[5, 2, 2],
              [4, 6, 1],
              [1, 3, 2],
              [2, 2, 3],
              [2,7,1]])
print(X.shape)


y = np.array([[1],
              [1],
              [0],
              [0],
              [0]])

np.random.seed(1)


syn0 = 2 * np.random.randn(3, 4) - 1
syn1 = 2 * np.random.randn(4, 1) - 1
syn2 = 2*np.random.rand(1,4)-1
syn3 = 2*np.random.rand(4,1)-1



for j in range(600000):

    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    l3 = nonlin(np.dot(l2,syn2))
    l4 = nonlin(np.dot(l3,syn3))
    # how much did we miss the target value?
    l4_error = y - l4

    if (j % 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l4_error))))
        print(l4)

    l4_delta = l4_error * nonlin(l4, deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l3_error = l4_delta.dot(syn3.T)

    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l3_delta = l3_error * nonlin(l3, deriv=True)

    l2_error=l3_delta.dot(syn2.T)
    l2_delta = l2_error*nonlin(l2,True)

    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error*nonlin(l1,deriv=True)

    syn3 += l3.T.dot(l4_delta)
    syn2 += l2.T.dot(l3_delta)
    syn1 += l1.T.dot(l2_delta)/9
    syn0 += l0.T.dot(l1_delta)

predict = np.array([5,3,1])
l0=predict
l1 = nonlin(np.dot(l0, syn0))
l2 = nonlin(np.dot(l1, syn1))
l3 = nonlin(np.dot(l2,syn2))
l4 = nonlin(np.dot(l3,syn3))
print(l4)
