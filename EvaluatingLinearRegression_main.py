import numpy as npy
import statistics as stats

file = open('X.csv','rt')
X = file.read()
X1 = X.split()
X = []
for element in X1:
    X.append(float(element))
file = open('y.csv','rt')
y = file.read()
y1 = y.split()
y = []
for element in y1:
    y.append(float(element))

file = open('Xval.csv','rt')
Xval = file.read()
Xval1 = Xval.split()
Xval = []
for element in Xval1:
    Xval.append(float(element))
file = open('yval.csv','rt')
yval = file.read()
yval1 = yval.split()
yval = []
for element in yval1:
    yval.append(float(element))

file = open('Xtest.csv','rt')
Xtest = file.read()
Xtest1 = Xtest.split()
Xtest = []
for element in Xtest1:
    Xtest.append(float(element))
file = open('ytest.csv','rt')
ytest = file.read()
ytest1 = ytest.split()
ytest = []
for element in ytest1:
    ytest.append(float(element))

def featureGenerator(X):
    d = 2 #degree of regression polynomial
    count = 0
    x_new = []
    for element in X:
        datapoint = [1]
        k = 1
        while k<=d:
            datapoint.append(npy.power(element, k))
            k += 1
        x_new.insert(count, datapoint)
        count += 1
    return x_new

def hypothesisFunc(theta, x1):
    h = 0
    for i in range(len(theta)):
        h += theta[i]*x1[i]
    return h

def costFunc(theta, x_norm, y, regParam):
    cost1 = 0
    for i in range(len(x_norm)):
        cost1 += npy.power((hypothesisFunc(theta, x_norm[i][:]) - y[i]), 2)
    cost2 = 0
    for i in range(len(theta)):
        cost2 += regParam*npy.power(theta[i], 2)
    cost = (cost1 + cost2)/(2*len(x_norm))
    return cost

def featureNormalizer(x_new):
    x_norm = x_new
    mu = []
    sigma = []
    for i in range(len(x_norm[0]) - 1):
        mu.append(sum(x_norm[:][i+1])/len(x_norm))
    for i in range(len(x_norm[0]) - 1):
        sigma.append(stats.stdev(x_norm[:][i+1]))
    for i in range(len(x_norm)):
        for j in range(len(x_norm[0]) - 1):
            x_norm[i][j+1] = (x_norm[i][j+1] - mu[j])/sigma[j]
    return x_norm

def gradientDescent(x_norm, y, theta, alpha, regParam):
    for i in range(maxIter):
        theta1 = theta
        for j in range(len(theta)):
            costDer = 0
            for k in range(len(x_norm)):
                costDer += (hypothesisFunc(theta, x_norm[k][:]) - y[k])*x_norm[k][j] + regParam*theta[j]
            theta1[j] = theta[j] - costDer*alpha/len(x_norm)
        theta = theta1
    return theta

alpha = 0.03
regParam = 3
maxIter = 1000
x_new = featureGenerator(X)
x_norm = featureNormalizer(x_new)
theta = npy.random.rand(len(x_norm[0]))
thetaTrained = gradientDescent(x_norm, y, theta, alpha, regParam)
print('Optimized parameters are:')
print(thetaTrained)
print('Error in trained set is:')
print(costFunc(thetaTrained, x_norm, y, regParam))

x_newCV = featureGenerator(Xval)
x_normCV = featureNormalizer(x_newCV)
costCV = costFunc(thetaTrained, x_normCV, yval, regParam)
print('Error in cross validation set is:')
print(costCV)

x_newTest = featureGenerator(Xtest)
x_normTest = featureNormalizer(x_newTest)
costTest = costFunc(thetaTrained, x_normTest, ytest, regParam)
print('Error in test set is:')
print(costTest)

