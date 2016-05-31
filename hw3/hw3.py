# -*- coding: utf-8 -*-

from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import det
import random




def distmat(X, Y):
    n = len(X)
    m = len(Y)
    xx = sum(X*X, axis=1)
    yy = sum(Y*Y, axis=1)
    xy = dot(X, Y.T)
    return tile(xx, (m, 1)).T+tile(yy, (n, 1)) - 2*xy


def calcProb(k,pMiu,pSigma):
    Px = zeros([N, k], dtype=float)
    for i in range(k):
        Xshift = mat(X - pMiu[i, :])
        inv_pSigma = mat(pSigma[:, :, i]).I
        coef = pow((2*pi), (len(X[0])/2)) * sqrt(det(mat(pSigma[:, :, i])))
        for j in range(N):
            tmp = (Xshift[j, :] * inv_pSigma * Xshift[j, :].T)
            Px[j, i] = 1.0 / coef * exp(-0.5*tmp)
    return Px

def genData(size):
    mean = [0,0]
    cov = [[1,0],[0,1]] 
    x,y = np.random.multivariate_normal(mean,cov,size).T
    return x, y

def plotResult(X, labels,iter):
    plt.figure();
    labels = array(labels).ravel()
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    


def MoG(X, k,  threshold=1e-15):
    N = len(X)
    labels = zeros(N, dtype=int)
    centers = array(random.sample(X, k))
    iter = 0
    
    # 所有分类高斯分布的中心
    pMiu = centers;
    # 任意一个点属于分类i的概率
    pPi = zeros([1,k], dtype=float) 
    # 所有分类的协方差矩阵
    pSigma = zeros([len(X[0]), len(X[0]), k], dtype=float)

    dist = distmat(X, centers)
    

    labels = dist.argmin(axis=1)

    for j in range(k):
        idx_j = (labels == j).nonzero()
        pMiu[j] = X[idx_j].mean(axis=0)
        pPi[0, j] = 1.0 * len(X[idx_j]) / N
        pSigma[:, :, j] = cov(mat(X[idx_j]).T)

    
    Lprev = -10000.0;
    
    pre_esp = 100000;
    
    while iter < 100:
        # 重新计算概率
        Px = calcProb(k,pMiu,pSigma)
        
        pGamma =mat(array(Px) * array(pPi))
        pGamma = pGamma / pGamma.sum(axis=1)

        Nk = pGamma.sum(axis=0) 
        
        # 调整分类的参数
        pMiu = diagflat(1/Nk) * pGamma.T * mat(X) 
        pPi = Nk / N #[1, K]
        pSigma = zeros([len(X[0]), len(X[0]), k], dtype=float)
        for j in range(k):
            Xshift = mat(X) - pMiu[j, :]
            for i in range(N):
                pSigmaK = Xshift[i, :].T * Xshift[i, :]
                pSigmaK = pSigmaK * pGamma[i, j] / Nk[0, j]
                pSigma[:, :, j] = pSigma[:, :, j] + pSigmaK
                
        #更新标签
        labels = pGamma.argmax(axis=1)
        
        # 检查收敛
        iter = iter + 1
        L = sum(log(mat(Px) * mat(pPi).T))
        cur_esp = L-Lprev
        if cur_esp < threshold:
            break
        if cur_esp > pre_esp:
            break
        pre_esp=cur_esp
        Lprev = L
        print "iter = %d, esp = %lf" % (iter,cur_esp)
    plotResult(X, labels,iter)

if __name__ == '__main__':
    X = np.matrix(genData(1000)).T;
    
    N = len(samples[0])
    X = zeros((N, 2))
    for i in range(N):
        #输出 samples[0][i]
        X[i, 0] = samples[0][i]
        X[i, 1] = samples[1][i]
    MoG(X, 3)

	
	