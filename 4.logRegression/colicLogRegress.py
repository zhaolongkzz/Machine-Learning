# -*- coding:utf-8 -*-
# !/usr/bin/python3.6

import numpy as np
import random, time

'''
当数据集较大时，我们使用改进的随机梯度上升算法
'''

def sigmoid(inX):
    '''
    Functions：
        sigmoid函数
    :return: 函数值
    Modify：
        2018-09-12

    如果在测试数据集中发现了一条数据的类别标签已经缺失，那么我们的简单做法是将该条数据丢弃。
    因为类别标签与特征不同，很难确定采用某个合适的值来替换。
    提示：RuntimeWarning: overflow encountered in exp return 1.0 / (1 + np.exp(-inX))
    说明计算的数据结果溢出了。虽然忽略这个报错也无妨。需要做以下调整，使用longfloat() 来解决溢出
    '''
    # return BigFloat.exact(1.0 / (1 + np.exp(-inX)))
    return 1.0 / (1 + np.exp(-inX))

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    '''
    Functions：
        改进的随机梯度上升算法
    :param dataMatrix: 数据数组
    :param classLabels: 数据标签
    :param numIter: 迭代次数
    :return:
        weights - 求得的回归系数数组(最优参数)
    Modify：
        2018-09-12
    '''
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights


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
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0;
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[-1]):
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