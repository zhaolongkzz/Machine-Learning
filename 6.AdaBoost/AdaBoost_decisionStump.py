# -*- coding:utf-8 -*-
# !/usr/bin/python3.6

'''
Before we build the algorithm of AdaBoost,
we should get the weak classifier which used with decision stump first.
'''

import numpy as np
import time
import matplotlib.pyplot as plt


def loadSimpData():
    '''
    创建单层决策树的数据集
    :return:
        dataMat - 数据矩阵
        classLabels - 数据标签
    '''
    datMat = np.matrix([[1., 2.1],
                        [1.5, 1.6],
                        [1.3, 1.],
                        [1., 1.],
                        [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def showDataSet(dataMat, labelMat):
    '''
    数据可视化
    :param dataMat: 数据矩阵
    :param labelMat: 数据标签
    :return: None
    '''
    data_plus = []                                                                  # 正样本
    data_minus = []                                                                 # 负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)                                              # 转换为numpy矩阵
    data_minus_np = np.array(data_minus)                                            # 转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])       # 正样本散点图，从列转为行
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])     # 负样本散点图
    plt.show()


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    '''
    单层决策树分类函数
    :param dataMatrix: 数据矩阵
    :param dimen: 第dimen列，也就是第几个特征
    :param threshVal: thresholdValue
    :param threshIneq: label
    :return:
        retArray - result of classifier
    '''
    # 初始化retArray为1，类似于标签的作用
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        # 如果小于阈值,则赋值为-1
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        # 如果大于阈值,则赋值为-1
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    '''
    找到数据集上最佳的单层决策树
    :param dataArr: 数据矩阵
    :param classLabels: 数据标签
    :param D: 样本权重
    :return:
        bestStump - 最佳单层决策树信息
        minError - 最小误差
        bestClasEst - 最佳的分类结果
    '''
    # 首要转mat，行矩阵和列矩阵
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    # 最小误差（最小错误率）初始化为正无穷大
    minError = float('inf')
    # 遍历所有的特征，对每个特征进行循环
    for i in range(n):
        # 找到特征中最小的值和最大值
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        # 计算步长
        stepSize = (rangeMax - rangeMin) / numSteps
        # 设置步长循环
        for j in range(-1, int(numSteps) + 1):
            # 对每个不等式循环
            for inequal in ['lt', 'gt']:  # 大于和小于的情况，均遍历。lt:less than，gt:greater than
                threshVal = (rangeMin + float(j) * stepSize)  # 计算阈值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)  # 计算分类结果
                errArr = np.mat(np.ones((m, 1)))  # 初始化误差矩阵
                errArr[predictedVals == labelMat] = 0  # 分类正确的,赋值为0
                weightedError = D.T * errArr  # 计算误差
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" %
                      (i, threshVal, inequal, weightedError))
                if weightedError < minError:  # 找到误差最小的分类方式
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


if __name__ == '__main__':
    dataArr, classLabels = loadSimpData()
    # showDataSet(dataArr, classLabels)
    D = np.mat(np.ones((5, 1)) / 5)
    bestStump, minError, bestClasEst = buildStump(dataArr, classLabels, D)
    print('bestStump:\n', bestStump)
    print('minError:\n', minError)
    print('bestClasEst:\n', bestClasEst)
