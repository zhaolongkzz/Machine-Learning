# -*- coding:utf-8 -*-
# !/usr/bin/python3.6

import operator
import numpy as np
from os import listdir
import KNN

def pic2Vector(filename):
    '''
    Function:
        将图像矩阵转化为向量形式
        目的是将每个文本的图像作为一个行向量，用于classify0()处理
    :param filename: 需要打开的文件
    :return:
        returnVect - 返回一个1*1024的向量
    Modify:
        2018-08-01
    '''
    returnVect = np.zeros((1, 1024))
    file = open(filename)
    for i in range(32):
        lineStr = file.readline()
        for j in range(32):
            # 每一行的32个元素依次添加到returnVect中，从0,32,64,96...依次开始添加
            returnVect[0, 32*i + j] = int(lineStr[j])
    return returnVect

def writingClassTest():
    # 测试集的Labels
    hwLabels = []
    # liststr得到该文件夹下的所有文件名（包括文件后缀）的一个列表
    trainingFileList = listdir('trainingDigits')
    # print(trainingFileList) 得到['0_0.txt', '0_1.txt',...]
    m = len(trainingFileList)
    # 初始化训练的Mat矩阵,测试集
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        # 获得文件的名字
        fileNameStr = trainingFileList[i]
        # 获得不同数字的分类，将列表根据'.'进行切片，得到单独文件名的列表
        classNum = int(fileNameStr.split('_')[0])
        # print(classNum) 得到 981 982 ...一系列的字符串
        hwLabels.append(classNum)     # 转回列表
        trainingMat[i, :] = pic2Vector('trainingDigits/%s' % (fileNameStr))

    # 返回testDigits目录下的文件名
    testFileList = listdir('testDigits')
    # 误差精度检测
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        # 获得文件的名字
        fileNameStr = testFileList[i]
        # 获得不同数字的分类，将列表根据'.'进行切片，得到单独文件名的列表
        classNum = int(fileNameStr.split('_')[0])
        # 将每一个文件的1x1024数据存储到trainingMat矩阵中
        vectorUnderTest = pic2Vector('testDigits/%s' % (fileNameStr))
        #获得预测结果
        classifierResult = KNN.classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNum))
        if(classifierResult != classNum):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest))


if __name__ == '__main__':
    # testVector = pic2Vector('testDigits/0_13.txt')
    # print(testVector[0, 32:63])
    writingClassTest()