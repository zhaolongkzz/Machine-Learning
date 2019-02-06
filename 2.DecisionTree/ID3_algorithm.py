# -*- coding:utf-8 -*-
# !/usr/bin/python3.6

from math import log
import operator
import conditionalEntropy as ce

def createDataSet():
    '''
    Function:
        创建测试数据集
    :return:
        dataSet - 数据集
        labels - 特征标签
    Modify:
        2018-08-12
    '''
    dataSet = [[0, 0, 0, 0, 'no'],                            #数据集
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
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']      # 特征标签
    return dataSet, labels                                    # 返回数据集和分类属性


def majorityCnt(classList):
    '''
    Function:
        统计classList中出现此处最多的元素(类标签)
    :param classList: 类标签列表
    :return:
        sortedClassCount[0][0] - 出现此处最多的元素(类标签)
    Modify:
        2018-08-12
    '''
    classCount = {}
    for vote in classList:                                    # 统计classList中每个元素出现的次数
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1                                 # 让新出现的类别数量归零，开始加一计数
    sortedClassCount = sorted(classCount.items(),             # 将字典中的keys和values组成一个元祖，并以列表的形式返回
                              key=operator.itemgetter(1),     # operator.itemgetter(1)函数是将对象的第1个域进行排序
                              reverse=True)                   # reverse=True降序，reverse=False升序（默认）
    return sortedClassCount[0][0]                             # 返回classList中出现次数最多的元素


def createTree(dataSet, labels, featLabels):
    '''
    Function:
        创建决策树
    :param dataSet: 训练数据集
    :param labels: 分类属性标签
    :param featLabels: 存储选择的最优特征标签
    :return:
        myTree - 决策树
    Modify:
        2018-08-12
    '''
    classList = [example[-1] for example in dataSet]          # 取分类标签(是否放贷:yes or no)
    # 递归的第一个停止条件是所有的类标签完全相同
    if classList.count(classList[0]) == len(classList):       # 如果类别完全相同则停止继续划分(统计字符出现的次数)
        return classList[0]
    # 递归的第二个停止条件是使用完了所有特征，其长度最终为1，则简单的返回唯一的类标签
    if len(dataSet[0]) == 1:                                  # 遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)

    bestFeat = ce.chooseBestFeatureToSplit(dataSet)           # 选择最优特征
    bestFeatLabel = labels[bestFeat]                          # 最优特征的标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel: {}}                              # 根据最优特征的标签生成树
    del (labels[bestFeat])                                    # 删除已经使用特征标签
    featValues = [example[bestFeat] for example in dataSet]   # 得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)                              # 去掉重复的属性值
    for value in uniqueVals:                                  # 遍历特征，创建决策树。
        myTree[bestFeatLabel][value] = \
            createTree(ce.splitDataSet(dataSet, bestFeat, value),
                       labels, featLabels)
    return myTree


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    featLabels = []
    myTree = createTree(dataSet, labels, featLabels)
    print(myTree)