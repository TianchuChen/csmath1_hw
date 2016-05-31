
from numpy import *
import numpy as np
import pylab as plt

def calcHAndP(D , sigma ):
    # 计算熵H和条件概率P
    P = np.exp(-D.copy() * sigma);
    sumP = sum(P);
    H = np.log(sumP) + sigma * np.sum(D * P) / sumP;
    P = P / sumP;
    return H, P;


def searchSigmaAndCalcP(X  , tol = 1e-5, perplexity = 30.0):
    # 根据输入的困惑度，通过二分搜索找到合适sigma并计算所有的条件概率Pi|j

    (n, d) = X.shape;
    sum_X = np.sum(np.square(X), 1);
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X);
    P = np.zeros((n, n));
    sigma = np.ones((n, 1));
    # 目标的熵值 H(P) = log(perp) 
    targetH = np.log(perplexity);

    # 遍历所有的Xi
    for i in range(1, n):

        sigmamin = -np.inf;
        sigmamax =  np.inf;
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))];
        (H, thisP) = calcHAndP(Di, sigma[i]);

        # 检查和目标的差距
        Hdiff = H - targetH;
        tries = 0;
        while np.abs(Hdiff) > tol and tries < 50:

            # 如果需要，重新计算sigma
            if Hdiff > 0:
                sigmamin = sigma[i].copy();
                if sigmamax == np.inf or sigmamax == -np.inf:
                    sigma[i] = sigma[i] * 2;
                else:
                    sigma[i] = (sigma[i] + sigmamax) / 2;
            else:
                sigmamax = sigma[i].copy();
                if sigmamin == np.inf or sigmamin == -np.inf:
                    sigma[i] = sigma[i] / 2;
                else:
                    sigma[i] = (sigma[i] + sigmamin) / 2;

            # 重新计算H和P
            (H, thisP) = calcHAndP(Di, sigma[i]);
            Hdiff = H - targetH;
            tries = tries + 1;

        # 设置所有的条件概率Pi|j
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP;


    return P;



def tsne(X , outputDims  , perp):

    # 对输入的NxD数据集进行 tsne 处理

    (n, d) = X.shape;
    max_iter = 1000;
    momentum = 0.5;
    eta = 300;
    Y = np.random.randn(n, outputDims);
    dY = np.zeros((n, outputDims));
    iY = np.zeros((n, outputDims));

    global P;

    P = searchSigmaAndCalcP(X, 1e-5, perp);

    # 转换成联合概率    
    P = P + np.transpose(P);
    
    # 由于要使用到K-L距离，确保概率和是1
    P = P / np.sum(P);               
    P = np.maximum(P, 1e-12);

    # 进行迭代
    for iter in range(0, max_iter):

        # 计算低维空间中的Qij
        sum_Y = np.sum(np.square(Y), 1);
        num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y));
        num[range(n), range(n)] = 0;
        Q = num / np.sum(num);
        Q = np.maximum(Q, 1e-12);

        # 计算梯度
        PQ = P - Q;
        for i in range(0, n):
            dY[i,:] = np.sum(np.tile(PQ[:,i] * num[:,i], (outputDims, 1)).T * (Y[i,:] - Y), 0);


        # 更新Y
        iY = momentum * iY - eta * ( dY);
        Y = Y + iY;
        Y = Y - np.tile(np.mean(Y, 0), (n, 1));

  
    return Y;


def genTestData():
    # 生成测试数据和对应的类别
    
    global X, labels;
    N = 10;
    D = 10;
    C = 10;
    X = np.zeros((N * C, D));
    labels = np.zeros(N*C);
    
    for i in range(0, C):
        
        # 数据集Di的各个维度通过平均值为i，方差为0.1的正态分布生成
        mean = np.full(D, i);
        cov = np.eye(D) * 0.1; 
        X[i * N: i * N + N, :] = np.random.multivariate_normal(mean,cov,N);
        labels[i*N: i*N + N] =  i;
    
        

if __name__ == "__main__":
    genTestData();
    Y = tsne(X, 2, 20.0);
    
    plt.figure(1);
    Y = Y.T;
    
    # 数据原本具有的类型会在图上以不同颜色显示
    plt.scatter(Y[0], Y[1], marker = 'o', c = labels, s = 10) 
    plt.show();
