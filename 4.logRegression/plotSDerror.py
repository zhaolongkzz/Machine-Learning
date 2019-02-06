# -*- coding:utf-8 -*-
# !/usr/bin/python3.6

from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import logRegress

def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.5
    weights = ones(n)   #initialize to all ones
    weightsHistory=zeros((500*m,n))
    for j in range(500):
        for i in range(m):
            h = logRegress.sigmoid(sum(dataMatrix[i]*weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[i]
            weightsHistory[j*m + i,:] = weights
    return weightsHistory


def stocGradAscent1(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.4
    weights = ones(n)   #initialize to all ones
    weightsHistory=zeros((40*m,n))
    for j in range(40):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = logRegress.sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            #print error
            weights = weights + alpha * error * dataMatrix[randIndex]
            weightsHistory[j*m + i,:] = weights
            del(dataIndex[randIndex])
    print(weights)
    return weightsHistory


def regPlot(myHist0, myHist1):
    n = shape(dataArr)[0] #number of points to create
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []

    markers =[]
    colors =[]

    fig = plt.figure()
    ax = fig.add_subplot(321)
    type1 = ax.plot(myHist0[:,0])
    plt.ylabel('X0')
    ax = fig.add_subplot(323)
    type1 = ax.plot(myHist0[:,1])
    plt.ylabel('X1')
    ax = fig.add_subplot(325)
    type1 = ax.plot(myHist0[:,2])
    plt.xlabel('iteration')
    plt.ylabel('X2')

    ax = fig.add_subplot(322)
    type1 = ax.plot(myHist1[:,0])
    plt.ylabel('X0')
    ax = fig.add_subplot(324)
    type1 = ax.plot(myHist1[:,1])
    plt.ylabel('X1')
    ax = fig.add_subplot(326)
    type1 = ax.plot(myHist1[:,2])
    plt.xlabel('iteration')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    dataMat, labelMat = logRegress.loadDataSet()
    dataArr = array(dataMat)
    myHist0 = stocGradAscent0(dataArr, labelMat)
    myHist1 = stocGradAscent1(dataArr, labelMat)
    regPlot(myHist0, myHist1)