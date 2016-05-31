# -*- coding: utf-8 -*-

from numpy import *
import numpy as np
import matplotlib.pyplot as plt


db = [];

def initData():
    x = []  
    currentNum = 0
    global db;
    
    
    with (open('data.txt')) as f:
        for line in (f.readlines()):
            line = line.strip();
            if len(line) < 5:
                currentNum = int(line);
                if (currentNum == 3):
                    db.append(x);
                x = [];
            else:
                for ch in line:
                    x.append(int(ch));
                    
def doPca():
    global center, comp, Y;
    
    mat = matrix(db).T
    center = mat - mat.mean(1)
    U, s, V = linalg.svd(center, full_matrices=True)
    u2 = matrix(array([(U[:,0]), (U[:,1])])).T
    Y = u2.getI()* center;
    
    
    
def drawResult():
    plt.figure(1);
    plt.scatter(Y[0], Y[1], marker = 'o', color = 'g', label='1', s = 30)  
    

initData();
doPca();
drawResult();