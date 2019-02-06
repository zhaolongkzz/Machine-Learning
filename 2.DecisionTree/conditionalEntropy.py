# -*- coding:utf-8 -*-
# !/usr/bin/python3.6

# 香农熵，条件熵，特征A对训练集D的信息增益g(D, A)
from math import log
import entropy as ent

def splitDataSet(dataSet, axis, value):
    '''
    Function:
        按照给定特征划分数据集
    :param dataSet: 待划分的数据集
    :param axis: 划分数据集的特征
    :param value: 需要返回特征的值
    :return: None
    Modify:
        2018-08-11
    '''
    # 创建返回的数据集列表
    retDataSet = []
    # 遍历数据集
    for featVec in  dataSet:
        if featVec[axis] == value:
            # 将axis特征前的特征赋值，不包括axis特征
            reducedFeatVec = featVec[:axis]
            # 将axis特征后的特征添加，上下两句去掉了axis特征
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    # 返回划分后的数据集
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    '''
    Function:
        选择最优特征
    :param dataSet: 数据集
    :return:
        bestFeature - 信息增益最大的(最优)特征的索引值
    Modify:
        2018-08-11
    '''
    # 取特征数量
    numFeatures = len(dataSet[0]) - 1
    # 计算数据集的香农熵
    baseEntropy = ent.calcShannonEnt(dataSet)
    # 信息增益
    bestInfoGain = 0.0
    # 最优特征的索引值
    baseFeature = -1
    # 遍历所有特征，i控制特征A的种类
    for i in range(numFeatures):
        # 获取dataSet的第i个特征的内容作为列表
        featList = [example[i] for example in dataSet]
        # 用set使featList中的元素不重复的展现出来
        uniqueVals = set(featList)
        # print(uniqueVals)为每个特征的种类{0, 1, 2}，{0, 1}，{0, 1}，{0, 1, 2}
        # 经验条件熵
        newEntropy = 0.0
        # 计算信息增益，value控制特征A中对应目标D的类别
        for value in uniqueVals:
            # 按照特征A(i)来划分数据集
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算A(i)子集的概率
            prob = len(subDataSet) / float(len(dataSet))
            # 根据公式计算经验条件熵
            newEntropy += prob * ent.calcShannonEnt(subDataSet)
        # 信息增益
        infoGain = baseEntropy - newEntropy
        # print('第%d个特征的增益为%.3f' % (i, infoGain))
        # 计算信息增益
        if (infoGain > bestInfoGain):
            # 更新信息增益，找到最大的信息增益
            bestInfoGain = infoGain
            # 记录信息增益最大的特征索引值
            bestFeature = i
        # 返回信息增益最大的特征的索引值
    return bestFeature


if __name__ == '__main__':
    dataSet, features = ent.createDataSet()
    print("最优特征索引值:" + str(chooseBestFeatureToSplit(dataSet)))