import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('iris.csv',names=["x1","x2","x3","x4","y"])
X = data.iloc[:,0:4]
y = data.iloc[:,4:5].values
print(X.shape)
print(y)

def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))



np.random.seed(1)


syn0 = 2 * np.random.randn(4, 4) - 1
syn1 = 2 * np.random.randn(4, 4) - 1
syn2 = 2*np.random.rand(4,4)-1
syn3 = 2*np.random.rand(4,1)-1

cost = np.zeros(100000)

for j in range(100000):


    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    l3 = nonlin(np.dot(l2,syn2))
    l4 = nonlin(np.dot(l3,syn3))


    l4_error = y - l4

    print(j,".Epoch ","Error:" + str(np.mean(np.abs(l4_error))))
    cost[j] = np.mean(np.abs(l4_error))

    l4_delta = l4_error * nonlin(l4, deriv=True)

    l3_error = l4_delta.dot(syn3.T)


    l3_delta = l3_error * nonlin(l3, deriv=True)

    l2_error=l3_delta.dot(syn2.T)
    l2_delta = l2_error*nonlin(l2,True)

    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error*nonlin(l1,deriv=True)

    syn3 += l3.T.dot(l4_delta)
    syn2 += l2.T.dot(l3_delta)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
print(l0.shape)
print(l1.shape)
print(l2.shape)
print(l3.shape)
print(l4.shape)

predict = np.array([[5.1,3.8,1.9,0.4]])
lx = predict
l1 = nonlin(np.dot(lx, syn0))
l2 = nonlin(np.dot(l1, syn1))
l3 = nonlin(np.dot(l2,syn2))
l4 = nonlin(np.dot(l3,syn3))
print(l4)

fig, ax = plt.subplots()
ax.plot(np.arange(100000), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()
