# -*- coding:utf-8 -*-
# !/usr/bin/python3.6

import numpy as np
import operator

import file2Matrix as f2m
import KNN

def classifyPerson():
    #输出结果
    resultList = {'C': '讨厌', 'B': '有些喜欢', 'A': '非常喜欢'}
    #三维特征用户输入
    precentTats = float(input("玩视频游戏所耗时间百分比:"))
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))

    #打开并处理数据
    datingDataMat, datingLabels = f2m.file2Matrix("testSet.txt")
    #训练集归一化
    normDataSet, ranges, minVals = f2m.autoNorm(datingDataMat)
    #生成NumPy数组,测试集
    inArr = np.array([ffMiles, precentTats, iceCream])
    #测试集归一化
    norminArr = (inArr - minVals) / ranges
    #返回分类结果
    classifierResult = KNN.classify0(norminArr, normDataSet, datingLabels, 3)
    #打印结果

    print("你可能%s这个人" % (resultList[classifierResult]))

if __name__ == '__main__':
    classifyPerson()