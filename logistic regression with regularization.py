import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def load_data (n):
    data=np.loadtxt("ex2data2.txt",dtype=np.float64, delimiter=",")
    feature = data [::,0:n]
    labels= data [:,n]
    return feature, labels

def plot_data (feature, label):
    for i in range (1,len(feature[:,0])):
        if (label[i]==0):
            plt.scatter(x=feature[i][0], y=feature[i][1], color='yellow', edgecolors='k', s=30)
        else:
            plt.scatter(x=feature[i][0], y=feature[i][1], marker='+', color='black')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.show()

def plot_data1 (label,u,v,z):
    data = np.loadtxt("ex2data2.txt", dtype=np.float64, delimiter=",")
    feature = data[::, 0:n]
    for i in range (1,len(feature[:,0])):
        if (label[i]==0):
            accepted = plt.scatter(x=feature[i][0], y=feature[i][1], color='yellow', edgecolors='k', s=30)
        else:
            rejected = plt.scatter(x=feature[i][0], y=feature[i][1], marker='+', color='black')
    plt.contour(u, v, z, 0)
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend((accepted, rejected), ('y=1', 'y=0'), loc='upper right')
    plt.show()


no_of_features=2
features, label= load_data (no_of_features)
plot_data (features,label)

def mapFeature(feature1, feature2):
    degree = 6
    out = np.ones(features.shape[0])[:,np.newaxis]
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(feature1, i-j),np.power(feature2, j))[:,np.newaxis]))
    return out

def mapFeature1(feature1, feature2):
    degree = 6
    out = np.ones(1)
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(feature1, i-j), np.power(feature2, j))))
    return out

def sigmoid(z):
  return 1/(1+np.exp(-z))

def CostFunc(theeta_t, feature_t, label_t, lambda_t):
    m = len(label_t)
    J = (-1/m) * (label_t.T @ np.log(sigmoid(feature_t @ theeta_t)) + (1 - label_t.T) @ np.log(1 - sigmoid(feature_t @ theeta_t)))
    reg = (lambda_t/(2*m)) * (theeta_t[1:].T @ theeta_t[1:])
    J = J + reg
    return J

def GradientDescent(theeta, feature, label, lambda_t):
    m = len(label)
    gradient = (1/m) * feature.T @ (sigmoid(feature @ theeta) - label)
    gradient[1:] = gradient[1:] + (lambda_t / m) * theeta[1:]
    return gradient

features = mapFeature(features[:,0], features[:,1])
(m, n) = features.shape
label = label[:, np.newaxis]
theetas = np.zeros((n,1))
lmbda = 1
J = CostFunc(theetas, features, label, lmbda)
print(J)
output = opt.fmin_tnc(func=CostFunc, x0=theetas.flatten(), fprime=GradientDescent, args=(features, label.flatten(), lmbda))
theetas = output[0]
print(theetas)
predict = [sigmoid(np.dot(features, theetas)) >= 0.5]
print(np.mean(predict == label.flatten()) * 100)
x = np.linspace(-1, 1.5, 50)
y = np.linspace(-1, 1.5, 50)
z = np.zeros((len(x), len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        z[i,j] = np.dot(mapFeature1(x[i], y[j]), theetas)
plot_data1 (label,x,y,z)