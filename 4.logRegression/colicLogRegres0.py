# -*- coding:utf-8 -*-
# !/usr/bin/python3.6

import numpy as np
import random, time

'''
当数据集较小时，我们使用梯度上升算法
'''

def sigmoid(inX):
    '''
    Functions：
        sigmoid函数
    :return: 函数值
    Modify：
        2018-09-12
    '''
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataMatIn, classLabels):
    '''
    Function:
        梯度上升算法
    :param dataMatIn: 数据集
    :param classLabels: 数据标签
    :return:
        weights.getA() - 求得的权重数组(最优参数)
    Modify:
        2018-09-12
    '''
    dataMatrix = np.mat(dataMatIn)  # 转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()  # 转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)  # 返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.01  # 移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500  # 最大迭代次数
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # 梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()  # 将矩阵转换为数组，并返回


def colicTest():
    '''
    Functions:
        使用Pyhton写的Logistic分类器做预测
    :return: None
    Modify：
        2018-09-12
    '''
    # 打开训练集，打开测试集
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = [];
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    # 使用改进的随即上升梯度训练
    trainWeights = gradAscent(np.array(trainingSet), trainingLabels)
    errorCount = 0;
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights[:, 0])) != int(currLine[-1]):
            errorCount += 1
    # 错误率计算
    errorRate = (float(errorCount) / numTestVec) * 100
    print("测试集错误率为: %.2f%%" % errorRate)


def classifyVector(inX, weights):
    '''
    Functions：
        分类函数
    :param inX: 特征向量
    :param weights: 回归系数
    :return: 分类结果
    Modify：
        2018-09-12
    '''
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


if __name__ == '__main__':
    start = time.clock()
    colicTest()
    end = time.clock()
    print('总耗时：' + str(end - start) + 's')