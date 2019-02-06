# -*- coding:utf-8 -*-
# !/usr/bin/python3.6

"""
Function:
    根据已知数据集来分析目标
Parameters:
    无
Returns:
    group - 数据集
    labels - 分类标签
Modify:
    2018-08-01
"""
import operator
import numpy as np

def createDataSet():
    #四组二维特征,array后等同于一个矩阵
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    # 四组特征的标签，类别，用于后续判断邻近坐标类别的数量
    labels = ['A','A','B','B']
    return group, labels


"""
Function:
    kNN算法,分类器
Parameters:
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labes - 分类标签
    k - kNN算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果
Modify:
    2018-08-01
"""
def classify0(inX, dataSet, labels, k):
    # 选取dataSet数据集中数组的行与列
    dataSetSize = dataSet.shape[0]
    # tile与matlab中的repmat(A, [m, n])函数类似，将矩阵A重复复制为m行n列的矩阵
    # 用于不同数据集中的元素与test元素进行比较
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 每个元素都平方
    sqDiffMat = diffMat**2
    # sum()函数为求和，其中axis=0为列相加，axis=1为行相加
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance**0.5
    # 矩阵中每个元素的坐标排序
    '''usage
    distance = np.array([2, 5, 3, 8])
    labels = ['A','A','B','B']
    classCount = {}
    dis = distance.argsort()
    print(dis)
    ==> [0 2 1 3]
    for i in range(3):
        k = labels[dis[i]]
        classCount[k] = classCount.get(k, 0) + 1
        print(classCount)
    ==> {'A': 1}
        {'A': 1, 'B': 1}
        {'A': 2, 'B': 1}
    '''
    sortedDistIndices = distance.argsort()
    # 给一个空的字典为了记录
    classCount = {}
    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        # 计算类别次数[ dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # python3中用items()替换python2中的iteritems()
        # key=operator.itemgetter(1)根据字典的值进行排序
        # key=operator.itemgetter(0)根据字典的键进行排序
        # reverse降序排序字典
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]


if __name__ == '__main__':
    # 创建数据集
    group, labels = createDataSet()
    # 测试集
    test = [10, 20]
    # kNN分类
    test_class = classify0(test, group, labels, 3)
    # 打印分类结果
    print(test_class)


