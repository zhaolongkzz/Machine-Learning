# -*- coding:utf-8 -*-
# !/usr/bin/python3.6

# 香农熵，所有类的信息期望值H
from math import log

def createDataSet():
    '''
    Function:
        创建测试数据集
    :return:
        dataSet - 数据集
        labels - 分类属性
    Modify:
        2018-08-10
    '''
    # 年龄：0代表青年，1代表中年，2代表老年；
    # 有工作：0代表否，1代表是；
    # 有自己的房子：0代表否，1代表是；
    # 信贷情况：0代表一般，1代表好，2代表非常好；
    # 类别(是否给贷款)：no代表否，yes代表是。
    dataSet = [[0, 0, 0, 0, 'no'],    #数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [0, 1, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 1, 1, 0, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 0, 1, 0, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['不放贷', '放贷']  # 分类属性
    return dataSet, labels  # 返回数据集和分类属性


def calcShannonEnt(dataSet):
    '''
    Function:
        计算给定数据集的经验熵(香农熵)
    :param dataSet: 数据集
    :return:
        shannonEnt - 经验熵(香农熵)
    Modify:
        2018-08-10
    '''
    # 返回数据集的行数
    numEntires = len(dataSet)
    # 保存每个标签 Label 出现次数的字典
    labelCounts = {}
    # 对每组特征向量进行统计
    for featVec in dataSet:
        # 提取标签数据集最后边一列的信息，作为标签
        # print(featVec)依次打印[0, 0, 0, 0, 'no']，[0, 0, 0, 1, 'no']...
        currentLabel = featVec[-1]
        # 开始no不在里面，使labelCounts[no] = 0开始计数，后续识别到no，+1；
        # 到yes则使labelCounts[yes] = 0开始计数
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # Label计数
        labelCounts[currentLabel] += 1
        # print(labelCounts)依次打印{'no': 1}，{'no': 2}，{'no': 2, 'yes': 1}...

    # 经验熵(香农熵)
    shannonEnt = 0.0
    # 计算香农熵
    for key in labelCounts:
        # labelCounts[key] = value次数，该Label的概率
        prob = float(labelCounts[key]) / numEntires
        # 利用公式计算
        shannonEnt -= prob * log(prob, 2)
    # 返回经验熵(香农熵)
    return shannonEnt


if __name__ == '__main__':
    dataSet, features = createDataSet()
    print(dataSet)
    print(calcShannonEnt(dataSet))