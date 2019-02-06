# -*- coding:utf-8 -*-
# !/usr/bin/python3.6

'''
Function:
    分类器测试函数
Parameters:
    None
Returns:
    normDataSet - 归一化后的特征矩阵
    ranges - 数据范围
    minVals - 数据最小值
Modify:
    2018-08-09
'''
import numpy as np
import operator

import file2Matrix as f2m
import KNN

def classTest():
    datingDataMat, datingLabels = f2m.file2Matrix("testSet.txt")
    # 获取数据的10%，保证精度，两位小数
    fetchData = 0.10
    # 归一化后得到normDataSet数据集矩阵
    normDataSet, ranges, minVals = f2m.autoNorm(datingDataMat)
    # 获取行数，shape[0]为行，sum(0)为列
    m = normDataSet.shape[0]
    # 行数 * 百分比 = 获取数据的行数
    numTestVecs = int(m * fetchData)
    # 分类时错误计数
    errorCount = 0.00
    for i in range(numTestVecs):
        # 该循环为前100行，normDataSet[i, :]表示矩阵的第i行的3个数据，后面的numTestVecs:m为训练集
        classifyResult = KNN.classify0(normDataSet[i, :], normDataSet[numTestVecs:m, :], datingLabels[numTestVecs:m], 4)
        print('分类测试的结果：%s\t真实的分类：%s' % (classifyResult, datingLabels[i]))
        if classifyResult != datingLabels[i]:
            errorCount += 1.00
    print('错误率：%f%%' % (errorCount / float(numTestVecs) * 100))

if __name__ == '__main__':
    classTest()