# -*- coding: utf-8 -*-
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

def LinearKernelFunc(x1, x2):
    return np.dot(x1,x2)

def GaussianKernelFunc(x1, x2, sigma):
    return np.exp(-np.power(np.linalg.norm(x1-x2),2)/(2*np.power(sigma,2)))

def PlotData(X, y, title=''):
    pos = np.flatnonzero(y>0)
    neg = np.flatnonzero(y<0)

    p1 = plt.plot(X[pos,0], X[pos,1], 'bo', lw=1, ms = 7)
    p2 = plt.plot(X[neg,0], X[neg,1], 'ro', lw=1, ms=7)
    plt.title(title)


class SVMClassifier(object):
    """A SVM classifier using a sipmlified version of the SMO algorithm."""

    def __init__(self, kernel='Linear', sigma=0.1):
        self.kernel = kernel
        self.sigma = sigma

    def train(self, X, Y, C=1, tol=1e-3, max_passes=5):

        # Data parameters
        m = X.shape[0]
        n = X.shape[1]

        # Variables
        alphas = np.zeros((m))
        b = 0
        E = np.zeros((m))
        passes = 0
        eta = 0
        L = 0
        H = 0

        # Pre-compute the Kernel Matrix since our dataset is small
        if self.kernel == 'Linear':
            K = np.dot(X,X.T)
        else:
            X2 = np.sum(np.power(X,2), axis=1)
            K = X2.T + (X2.T - 2*np.dot(X,X.T)).T
            K = K.T*GaussianKernelFunc(1,0,self.sigma)


        # Train
        while passes < max_passes:
            num_changed_alphas = 0
            for i in range(m):
                E[i] = b + np.sum(alphas*Y*K[:,i]) - Y[i]

                if (Y[i]*E[i]<-tol and alphas[i]<C) or (Y[i]*E[i]>tol and alphas[i]>0):
                    j = np.random.randint(0,m)
                    # Make sure i neq j
                    while j==i:
                        j = np.random.randint(0,m)

                    E[j] = b + np.sum(alphas*Y*K[:,j]) - Y[j]

                    # Save old alphas
                    alpha_i_old = alphas[i]
                    alpha_j_old = alphas[j]

                    # Compute L and H
                    if Y[i] == Y[j]:
                        L = max(0, alphas[j]+alphas[i]-C)
                        H = min(C, alphas[j]+alphas[i])
                    else:
                        L = max(0, alphas[j]-alphas[i])
                        H = min(C, C+alphas[j]-alphas[i])

                    if L == H:
                        continue

                    # Compute eta
                    eta = 2 * K[i,j] - K[i,i] - K[j,j]
                    if eta >= 0:
                        continue

                    # Compute and clip new value for alpha j
                    alphas[j] = alphas[j] - (Y[j]*(E[i]-E[j]))/eta

                    # Clip
                    alphas[j] = min(H, alphas[j])
                    alphas[j] = max(L, alphas[j])

                    # Check if change in alpha is significant
                    if abs(alphas[j]) - alpha_j_old < tol:
                        # continue to next i
                        # replace anyway
                        alphas[j] = alpha_j_old
                        continue

                    # Determine value for alpha i
                    alphas[i] = alphas[i] + Y[i]*Y[j]*(alpha_j_old - alphas[j])

                    # Compute b1 and b2
                    b1 = b - E[i] \
                        - Y[i] * (alphas[i] - alpha_i_old) * K[i,j] \
                        - Y[j] * (alphas[j] - alpha_j_old) * K[i,j]
                    b2 = b - E[j] \
                        - Y[i] * (alphas[i] - alpha_i_old) * K[i,j] \
                        - Y[j] * (alphas[j] - alpha_j_old) * K[j,j]

                    # Compute b
                    if 0 < alphas[i] and alphas[i] < C:
                        b = b1
                    elif 0 < alphas[j] and alphas[j] < C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2

                    num_changed_alphas = num_changed_alphas + 1

            if num_changed_alphas == 0:
                passes = passes + 1
            else:
                passes = 0

        idx = (alphas > 0)
        self.X = X[idx,:]
        self.y = Y[idx]
        self.b = b
        self.alphas = alphas[idx]
        self.w = np.dot(alphas*Y,X)

    def predict(self, X):
        m = X.shape[0]
        p = np.zeros((m))
        pred = np.zeros((m))

        if self.kernel == 'Linear':
            p = np.dot(X,self.w) + self.b
        else:
            X1 = np.sum(np.power(X,2),axis=1)
            X2 = np.sum(np.power(self.X,2),axis=1)
            K = (X1 + (X2 - 2*np.dot(X,self.X.T)).T).T
            K = K * GaussianKernelFunc(1,0,self.sigma)
            K = self.y * K
            K = self.alphas * K
            p = np.sum(K,axis=1)

        pred[p>=0] = 1
        pred[p<0] = -1

        return pred

    def visualizeBoundary(self, X, y, title=''):
        PlotData(X, y, title)
        if self.kernel == 'Gaussian':
            x1plot = np.linspace(min(X[:,0]), max(X[:,0]), 100)
            x2plot = np.linspace(min(X[:,1]), max(X[:,1]), 100)
            [X1, X2] = np.meshgrid(x1plot, x2plot)
            vals = np.zeros((X1.shape[0],X1.shape[1]))
            for i in range(X1.shape[1]):
                this_X = np.c_[X1[:,i], X2[:,i]]
                vals[:,i] = self.predict(this_X)

            plt.contour(X1, X2, vals, [0, 0], colors='r')
        else:
            w = self.w
            b = self.b
            global xp;
            xp = np.linspace(min(X[:,0]), max(X[:,0]), 100)
            
            yp = - (w[0]*xp + b) / w[1]
            plt.plot(xp, yp,'r-')
            

def initData():
    global X, Y;
    X = [];
    Y = [];
    with(open('svmdata', 'r')) as f:
        for line in f.readlines():
            arr = line.strip().split(',');
            if (len(arr) < 3):
                continue;
            X.append([int(arr[0]), int(arr[1])]);
            y = 1;
            if (arr[2].strip() == 'b'):
                y = -1;
            Y.append(y);
    X = np.array(X);
    Y = np.array(Y);
    
        

initData();



# Seperate svm-data into training and test set
plt.figure();

svmlinear = SVMClassifier(kernel='Linear')
Y = Y.T;
svmlinear.train(X, Y)
svmlinear.visualizeBoundary(X, Y)
plt.show()
