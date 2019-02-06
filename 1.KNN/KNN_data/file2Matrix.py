# -*- coding:utf-8 -*-
# !/usr/bin/python3.6

"""
Function:收集数据
    打开文件并解析文件，对数据进行分类：
    C代表不喜欢,B代表魅力一般,A代表极具魅力
Parameters:
    filename - 文件名
Returns:
    returnMat - 特征矩阵
    classLabelVector - 分类Label向量
Modify:
    2018-08-08
"""
import os
import numpy as np

def file2Matrix(filename):
    # 打开数据集的文件
    file = open(filename)
    # 读取text文本中所有内容，以计算机的方式展现
    arrayLines = file.readlines()
    # 得到文件的行数
    numLines = len(arrayLines)
    # 返回的NumPy矩阵,解析完成的数据:numberOfLines行,3列
    returnMat = np.zeros((numLines, 3))
    # 定义：返回的分类标签向量
    classLabelVector = []

    index = 0
    for line in arrayLines:
        # s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        # 使用s.split(str="",num=string,cout(str))，将字符串根据'\t'分隔符进行切片，得到一个元素列表
        listOfline = line.split('\t')
        # 将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        returnMat[index, :] = listOfline[0:3]
        # 索引值-1可以表示为列表中的最后一个元素
        # 书中classLabelVector.append(int(listOfline[-1]))的int无法进行转换
        if listOfline[-1] == 'didntLike':
            classLabelVector.append('C')
        elif listOfline[-1] == 'smallDoses':
            classLabelVector.append('B')
        elif listOfline[-1] == 'largeDoses':
            classLabelVector.append('A')
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    """
    Function:
        归一化处理
    Parameters:
        dataSet - 待处理的数据集
    Returns:
        normDataSet - 归一化后的数据集
        ranges - 数据范围
        minVals - 最小值
    Modify:
        2018-08-08
    """
    # max,min和sum()一样，0为列，1为行
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    # shape()得到数据集的行与列
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


if __name__ == '__main__':

    filename = "testSet.txt"
    datingDataMat, datingLabels = file2Matrix(filename)
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    print(normDataSet)
    print(datingLabels)
