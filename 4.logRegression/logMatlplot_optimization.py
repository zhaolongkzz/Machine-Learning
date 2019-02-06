# -*- coding:utf-8 -*-
# !/usr/bin/python3.6

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import random

def loadDataSet():
    '''
    Function:
        加载数据
    :return:
        dataMat - 数据列表
        labelMat - 标签列表
    Modify:
        2018-09-09
    '''
    dataMat = []                                                            # 创建数据列表
    labelMat = []                                                           # 创建标签列表
    fr = open('testSet.txt')                                                # 打开文件
    for line in fr.readlines():                                             # 逐行读取
        lineArr = line.strip().split()                                      # 去回车，放入列表
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])         # 添加数据
        labelMat.append(int(lineArr[2]))                                    # 添加标签
    fr.close()                                                              # 关闭文件
    return dataMat, labelMat                                                # 返回


def sigmoid(inX):
    '''
    Function:
        sigmoid函数
    :return:
        dataMat - 数据列表
        labelMat - 标签列表
    :param inX:数据
    :return:
        sigmoid函数
    Modify:
        2018-09-09
    '''
    return 1.0 / (1 + np.exp(-inX))


def plotBestFit(weights):
    '''
    Function:
        绘制数据集
    :param weights: 权重参数数组
    :return: None
    Modify:
        2018-09-09
    '''
    dataMat, labelMat = loadDataSet()                                       # 加载数据集
    dataArr = np.array(dataMat)                                             # 转换成numpy的array数组
    n = np.shape(dataMat)[0]                                                # 数据个数
    xcord1 = [];
    ycord1 = []                                                             # 正样本
    xcord2 = [];
    ycord2 = []                                                             # 负样本
    for i in range(n):                                                      # 根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]);
            ycord1.append(dataArr[i, 2])                                    # 1为正样本
        else:
            xcord2.append(dataArr[i, 1]);
            ycord2.append(dataArr[i, 2])                                    # 0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)                                               # 添加subplot
    # 绘制的时候xcord1为一个列表保存所有的x
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)         # 绘制正样本
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)                   # 绘制负样本
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.title('BestFit')                                                    # 绘制title
    plt.xlabel('X1');
    plt.ylabel('X2')                                                        # 绘制label
    plt.show()


def stocGradAscent0(dataMatrix, classLabels):
    '''
    Function:
        随机梯度上升算法
    :param dataMatrix: 数据数组
    :param classLabels: 数据标签
    :return:
        weights - 求得的回归系数数组(最优参数)
    Modify:
        2018-09-09
    '''
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        # 随机梯度上升算法的h和error为向量，需要进行一些矩阵运算
        # 之前的方法为numpy的数组。以下的sum取矩阵中数的和
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    '''
    Function:
        优化的随机梯度上升算法
    :param dataMatrix: 数据数组
    :param classLabels: 数据标签
    :param numIter: 迭代次数
    :return:
        weights - 求得的回归系数数组(最优参数)
    Modify:
        2018-09-09
    '''
    m, n = np.shape(dataMatrix)                                             # 返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)                                                    # 参数初始化
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # alpha每次迭代的过程中都在调整，可以缓解数据波动
            alpha = 4 / (1.0 + j + i) + 0.01                                # 降低alpha的大小，每次减小1/(j+i)。
                                                                            # 当j<<max(i)时，alpha就不是严格下降的
            randIndex = int(random.uniform(0, len(dataIndex)))              # 随机选取样本，更新回归系数，减少周期性的波动
            h = sigmoid(sum(dataMatrix[randIndex] * weights))               # 选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h                              # 计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]       # 更新回归系数
            del (dataIndex[randIndex])                                      # 删除已经使用的样本
    return weights


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights = stocGradAscent1(np.array(dataMat), labelMat, 500)             # 此处的值不设置即为默认
    stocGradAscent0(np.array(dataMat), labelMat)
    plotBestFit(weights)