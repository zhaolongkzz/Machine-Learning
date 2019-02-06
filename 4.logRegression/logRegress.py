# -*- coding:utf-8 -*-
# !/usr/bin/python3.6

import numpy as np
import matplotlib.pyplot as plt

def Gradient_Ascent_test():
    '''
    Function:
        梯度上升算法测试函数
        求函数f(x) = -x^2 + 4x的极大值
    :return: None
    Modify:
        2018-08-16
    '''
    def f_prime(x_old):                                 # 函数f(x) = -x^2 + 4x的倒数
        return -2 * x_old +4
    x_old = -1                                          # 初始值，给一个小于x_new的值
    x_new = 0                                           # 梯度上升算法初始值，即从(0,0)开始
    alpha = 0.01                                        # 步长，也就是学习速率，控制更新的幅度
    presision = 0.00000001                              # 精度，也就是更新阈值
    while abs(x_new - x_old) > presision:
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)          # 梯度上升法
    print(x_new)


def sigmoid(inX):
    '''
    Function:
        创建sigmoid函数
    :param inX:
    :return:
    Modify:
        2018-08-16
    '''
    return 1.0 / (1 + np.exp(-inX))


def loadDataSet():
    '''
    Function:
        加载数据
    :return:
        dataMat - 数据列表
        labelMat - 标签列表
    Modify:
        2018-08-16
    '''
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    # 逐行读取
    for line in fr.readlines():
        # lineArr = line.split()可以直接划分字符串了
        lineArr = line.strip().split()
        # 将列表对象添加到dataMat中，注意为多精度的小数float
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        # 同理将最后一列的标签信息添加，按照整型添加
        labelMat.append(int(lineArr[-1]))
    return dataMat, labelMat


def gradAscent(dataMatIn, classLabels):
    '''
    Function:
        梯度上升算法
    :param dataMatIn: 数据集
    :param classLabels: 数据标签
    :return:
        weights.getA() - 求得的权重数组(最优参数)
    Modify:
        2018-08-16
    '''
    # 将dataMatIn列表信息转化为numpy矩阵
    dataMatrix = np.mat(dataMatIn)
    # 转换成numpy的mat,并进行转置
    labelMat = np.mat(classLabels).transpose()
    # 输出dataMatrix矩阵的行与列
    m, n = np.shape(dataMatrix)
    # 移动步长,也就是学习速率,控制更新的幅度
    alpha = 0.001
    # 最大迭代次数
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        # 梯度上升矢量化公式
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    # 将矩阵转换为数组，返回权重数组
    # mat.getA() 为自身矩阵变量转化为array类型的变量  [[ 4.12414349] [ 0.48007329] [-0.6168482 ]] class 'numpy.ndarray'
    # mat.getA1() 为将自身矩阵变换为一维的array类型   [ 4.12414349  0.48007329 -0.6168482 ]   class 'numpy.ndarray'
    # mat.getH() 返回自身(如果是复数矩阵)对偶转置矩阵，[[ 4.12414349  0.48007329 -0.6168482 ]] class 'numpy.matrixlib.defmatrix.matrix'
    # 如果为实数矩阵，则等价于np.transpose(self)
    # mat.getI() 返回可逆矩阵的逆                    [[ 0.23406658  0.02724665 -0.03500934]]
    # 如果直接用mat的返回值，其为 class 'numpy.matrixlib.defmatrix.matrix' 矩阵类型
    return weights.getA()


if __name__ == '__main__':
    Gradient_Ascent_test()
    dataMat, labelMat = loadDataSet()
    print(gradAscent(dataMat, labelMat))