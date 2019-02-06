# -*- coding:utf-8 -*-
# !/usr/bin/python3.6

import numpy as np
import matplotlib.pyplot as plt
import logRegress as lr

def loadDataSet():
    '''
    Function:
        加载数据
    :return
        dataMat - 数据列表
        labelMat - 标签列表
    Modify:
        2018-08-16
    '''
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        # lineArr = line.split()可以直接划分字符串了
        lineArr = line.strip().split()
        # 将列表对象添加到dataMat中，注意为多精度的小数float
        # 为了方便计算，该函数还将X0的值设为1.0
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        # 同理将最后一列的标签信息添加，按照整型添加
        labelMat.append(int(lineArr[-1]))
    return dataMat, labelMat


def plotDataSet():
    '''
    Function:
        加载数据
    :return: None
    Modify:
        2018-08-16
    '''
    dataMat, labelMat = loadDataSet()                                       # 加载数据集
    dataArr = np.array(dataMat)                                             # 转换成numpy的array数组
    n = np.shape(dataMat)[0]                                                # 数据数组的个数
    xcord1 = []; ycord1 = []                                                # 正样本
    xcord2 = []; ycord2 = []                                                # 负样本
    for i in range(n):                                                      # 根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])        # 1为正样本
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])        # 0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)                                               # 添加subplot
    # scatter函数中x，y为二维坐标，s为size，c为color，marker为形状，alpha为透明度的值
    ax.scatter(xcord1, ycord1, s= 20, c= 'red', marker= 's', alpha=.5)      # 绘制正样本
    ax.scatter(xcord2, ycord2, s= 20, c= 'green', alpha=.5)                 # 绘制负样本,marker默认是圆点

    plt.title('DataSet')                                                    # 绘制title
    plt.xlabel('x'); plt.ylabel('y')                                        # 绘制label
    plt.show()


def plotBestFit(weights):
    '''
    Function:
        绘制数据集
    :param weights: 权重参数数组
    :return: None
    Modify:
        2018-08-16
    '''
    dataMat, labelMat = loadDataSet()                                       # 加载数据集
    dataArr = np.array(dataMat)                                             # 转换成numpy的array数组
    n = np.shape(dataMat)[0]                                                # 数据个数
    xcord1 = []; ycord1 = []                                                # 正样本
    xcord2 = []; ycord2 = []                                                # 负样本
    for i in range(n):                                                      # 根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])      # 1为正样本
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])      # 0为负样本

    fig = plt.figure()
    ax = fig.add_subplot(111)                                               # 添加subplot
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='*', alpha=.5)         # 绘制正样本
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)                   # 绘制负样本,marker默认是圆点
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)

    plt.title('BestFit')                                                    # 绘制title
    plt.xlabel('X1');
    plt.ylabel('X2')                                                        # 绘制label
    plt.show()


if __name__ == '__main__':
    plotDataSet()
    dataMat, labelMat = lr.loadDataSet()
    weights = lr.gradAscent(dataMat, labelMat)
    plotBestFit(weights)
