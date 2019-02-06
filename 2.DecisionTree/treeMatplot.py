# -*- coding:utf-8 -*-
# !/usr/bin/python3.6

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator
import ID3_algorithm as ID3

def getNumLeafs(myTree):
    '''
    Function:
        获取决策树叶子结点的数目
    :param myTree: 决策树
    :return:
        numLeafs - 决策树的叶子结点的数目
    Modify:
        2018-08-12
    '''
    # 初始化叶子
    numLeafs = 0
    # python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    firstStr = next(iter(myTree))
    # 获取下一组字典
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    '''
    Function:
        获取决策树的层数
    :param myTree: 决策树
    :return:
        maxDepth - 决策树的层数
    Modify:
        2018-08-12
    '''
    # 初始化决策树深度
    maxDepth = 0
    # python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    firstStr = next(iter(myTree))
    # 获取下一个字典
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        # 更新层数
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    '''
    Function:
        绘制结点
    :param nodeTxt: 结点名
    :param centerPt: 文本位置
    :param parentPt: 标注的箭头位置
    :param nodeType: 结点格式
    :return: None
    Modify:
        2018-08-12
    '''
    # 定义箭头格式
    arrow_args = dict(arrowstyle="<-")
    # 设置中文字体
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 绘制结点
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType,
                            arrowprops=arrow_args, FontProperties=font)


def plotMidText(cntrPt, parentPt, txtString):
    '''
    Function:
        标注有向边属性值
    :param cntrPt: 用于计算标注位置
    :param parentPt: 用于计算标注位置
    :param txtString: 标注的内容
    :return: None
    Modify:
        2018-08-12
    '''
    # 计算标注位置
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    '''
    Function:
        绘制决策树
    :param myTree: 决策树(字典)
    :param parentPt: 标注的内容
    :param nodeTxt: 结点名
    :return: None
    Modify:
        2018-08-12
    '''
    # 设置结点格式
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    # 设置叶结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")
    # 获取决策树叶结点数目，决定了树的宽度
    numLeafs = getNumLeafs(myTree)
    # 获取决策树层数
    depth = getTreeDepth(myTree)
    # 下个字典
    firstStr = next(iter(myTree))
    # 中心位置
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    # 标注有向边属性值
    plotMidText(cntrPt, parentPt, nodeTxt)
    # 绘制结点
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]  # 下一个字典，也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  # y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            plotTree(secondDict[key], cntrPt, str(key))  # 不是叶结点，递归调用继续绘制
        else:  # 如果是叶结点，绘制叶结点，并标注有向边属性值
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    '''
    Function:
        创建绘制面板
    :param inTree: 决策树(字典)
    :return: None
    Modify:
        2018-08-12
    '''
    fig = plt.figure(1, facecolor='white')  # 创建fig
    fig.clf()  # 清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))  # 获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))  # 获取决策树层数
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0;  # x偏移
    plotTree(inTree, (0.5, 1.0), '')  # 绘制决策树
    plt.show()  # 显示绘制结果


if __name__ == '__main__':
    dataSet, labels = ID3.createDataSet()
    featLabels = []
    myTree = ID3.createTree(dataSet, labels, featLabels)
    print(myTree)
    createPlot(myTree)