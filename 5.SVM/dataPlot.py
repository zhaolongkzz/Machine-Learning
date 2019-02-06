# -*- coding:utf-8 -*-
# !/usr/bin/python3.6

import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(fileName):
    '''
    Functions:
        读取数据
    :param fileName: 文件名
    :return:
        dataMat - 数据矩阵
        labelMat - 数据标签
    Modify:
        2018-09-13
    '''
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[-1]))
    # n = len(labelMat)
    # for i in range(n):
    #     if labelMat[i] == 0.0:
    #         labelMat[i] = -1.0
    return dataMat, labelMat


def showDataSet(dataMat, labelMat):
    '''
    Functions:
        数据可视化
    :param dataMat: 数据矩阵
    :param labelMat: 数据标签
    :return: None
    Modify:
        2018-09-13
    '''
    # ********以下按照之前学习的绘图方法进行绘制**********
    data_plus = []          # 正样本
    data_minus = []         # 负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    # 将列表转化为numpy矩阵，有利于行列抓换
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    # 将numpy矩阵的行列转换，scatter(x, y)中的x和y只能识别单一的列表
    # 即：x = [3, 4, 5...]，而x = [[3, 4], [4, 6]...]则无法进行辨识
    data_plus_np = np.transpose(data_plus)
    data_minus_np = np.transpose(data_minus)

    ax.scatter(data_plus_np[0], data_plus_np[1], s= 20, c= 'blue', marker= 's', alpha= .5)
    ax.scatter(data_minus_np[0], data_minus_np[1], s= 20, c= 'orange', alpha= .5)
    plt.show()

    # data_plus = []  # 正样本
    # data_minus = []  # 负样本
    # for i in range(len(dataMat)):
    #     if labelMat[i] > 0:
    #         data_plus.append(dataMat[i])
    #     else:
    #         data_minus.append(dataMat[i])
    # data_plus_np = np.array(data_plus)  # 转换为numpy矩阵
    # data_minus_np = np.array(data_minus)  # 转换为numpy矩阵
    # plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])  # 正样本散点图
    # plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])  # 负样本散点图
    # plt.show()


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('testSet.txt')
    showDataSet(dataMat, labelMat)
