# -*- coding:utf-8 -*-
# !/usr/bin/python3.6

from math import log
import operator
import ID3_algorithm as ID3

def classify(inputTree, featLabels, testVec):
    '''
    Function:
        使用决策树分类
    :param inputTree: 已经生成的决策树
    :param featLabels: 存储选择的最优特征标签
    :param testVec: 测试数据列表，顺序对应最优特征标签
    :return:
        classLabel - 分类结果
    Modify:
        2018-08-12
    '''
    # 获取决策树结点
    firstStr = next(iter(inputTree))
    # 下一个字典
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


if __name__ == '__main__':
    dataSet, labels = ID3.createDataSet()
    featLabels = []
    myTree = ID3.createTree(dataSet, labels, featLabels)
    # 测试数据
    testVec = [0, 1, 0]
    result = classify(myTree, featLabels, testVec)
    if result == 'yes':
        print('放贷')
    if result == 'no':
        print('不放贷')