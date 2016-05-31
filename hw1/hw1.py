# -*- coding: utf-8 -*-

from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing


N = 100
M = 9
lam =  0# 1e-2 # 设置为非0的时使用这个变量的值作为 regularization term


# 产生带有噪音的 y
def mkdata(x):
	y = sin(x)
	y = [i+random.normal(0,0.1) for i in y]
	return y;


xS=linspace(0,6,100)
yS = sin(xS);

xN = linspace(0, 6, N);
yN = mkdata(xN);

# 用于计算 X, X^1 ... X^N （没有包含 bias）
poly = preprocessing.PolynomialFeatures(M, include_bias = False)


if (lam != 0):
        clf = linear_model.Ridge();
        clf.set_params(alpha=lam)
else:
        clf = linear_model.LinearRegression()


clf.fit(poly.fit_transform(matrix(xN).transpose()), yN);

yP = clf.predict(poly.fit_transform(matrix(xS).transpose()))

plt.figure(1);
plt.plot( xS, yS, "r-", label='sin(x)', linewidth=3)
plt.plot( xS, yP, 'b-', label='fit', linewidth = 3);
for i in range(1, N):
        plt.plot(xN, yN, 'bo');
plt.show();
