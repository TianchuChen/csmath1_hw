# -*- coding: utf-8 -*-

from numpy import *
import numpy as np
import matplotlib.pyplot as plt


e = 0.00000001
ie = 1.0 / e


#  x^2 + (y-1) ^2 ，这个函数在 X=0，Y=1时有最小值
def funcxy(X,Y):
    return np.power(X, 2)+ np.power((Y -1), 2)
    

def func(X):
    return funcxy(X[0],X[1])
    
# 计算一阶导数
def fderiv(X,i):#
    if i==0:
        return 2*X[0]
    elif i==1:
        return 2 *(X[1] - 1)
    else:
        return 0.0
        
# 计算二阶导数
def fderiv2(X,i,j):
    if i==0 and j==0:
        return 2
    elif i==0 and j==1:
        return 0
    elif i==1 and j==0:
        return 0
    elif i==1 and j==1:
        return 2
    else:
        return 0
        


def positive_definite(M):
    w, v = linalg.eig(M)
    t = w.min()
    if t >= 0: return 1
    else: return 0
    
def calg(X):
    n = X.size
    g = zeros(n)
    g.shape = (n, 1)
    for i in range(X.size):
        g[i] = fderiv(X, i)
    return g

def calG(X):
    n = X.size
    G = mat(arange(0.0, 1.0 * n * n, 1))
    G.shape = (n, n)
    for i in range(n):
        for j in range(n):
            G[i, j] = fderiv2(X, i, j)
    return G

def LMAlgorithm():
            
    xi = 1.0
    yi = 0.5
    u = 1.0

    x = array([xi, yi])
    x = x.reshape(x.size, 1)

    f = func(x)
    g = calg(x)
    G = calG(x)
    
    iter = 0;

    while vdot(g, g) >= e * e:
        G2 = G.copy()
        while positive_definite(G2 + identity(x.size) * u) == 0:
            u *= 4
        G2 += identity(x.size) * u

        A = mat(G2)
        B = mat(-g)
        s = linalg.solve(A, B)

        f2 = func(x + s)
        df = f2 - f
        dq = dot(g.T, s) + 0.5 * dot(dot(s.T, G), s)
        rk = dq / df
        if rk < 0.25:
            u *= 4
        elif rk > 0.75:
            u *= 0.5
        if rk > 0.0:
            xi2 = (x[0] + s[0])
            yi2 = (x[1] + s[1])
            x2 = array([xi2, yi2])
            x2 = x2.reshape(x2.size, 1)
            
            x = x2
            f = f2
        g = calg(x)
        G = calG(x)
        iter += 1;
        print 'iter %d: %.5f, %.5f'%(iter, x[0], x[1])
    print 'final : %.5f, %.5f'%(x[0], x[1])

    

if __name__ == "__main__":
    LMAlgorithm()
        


