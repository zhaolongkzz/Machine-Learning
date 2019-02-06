# -*- coding:utf-8 -*-
# !/usr/bin/python3.6

import numpy as np
from functools import reduce

def loadDataSet():
    '''
    Function:
        创建实验样本
    :return:
        postingList - 实验样本切分的词条
        classVec - 类别标签向量
    Modify:
        2018-08-14
    '''
    # 切分的词条
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 类别标签向量，1代表侮辱性词汇，0代表不是
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocabList(dataSet):
    '''
    Function:
        将切分的实验样本词条整理成不重复的词条列表
    :param dataSet: 整理的样本数据集
    :return:
        vocabSet:返回不重复的词条列表，即词汇表
    Modify:
        2018-08-14
    '''
    # 创建一个空的不重复的列表
    vocabSet = set([])
    for document in dataSet:
        # 取并集，为了使covabSet中的元素不重复
        # 首次vocabSet为0故并的值就是document，但是顺序随机变化了
        # 求并集的两个目标需要为set类型
        vocabSet = vocabSet | set(document)
    # 返回不重复的原数据集
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    '''
    Function:
        inputSet的元素对比于vocabList词汇表，
        将inputSet向量化，全部化为 1或 0
    :param vocabList: createVocabList返回的列表
    :param inputSet: 切分的词条列表
    :return:
        returnVec - 文档向量,词集模型
    Modify:
        2018-08-14
    '''
    # 创建一个根据len()长度，其中元素都为0的向量
    returnVec = [0] * len(vocabList)
    # 遍历每个词条
    for word in inputSet:
        # 如果词条存在于列表中，则置为 1
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: %s is not my Vocabulary!' % word)
    # 返回向量文档
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    '''
    Function:
        朴素贝叶斯分类器训练函数
    :param trainMatrix: 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
    :param trainCategory: 训练类别标签向量，即loadDataSet返回的classVec
    :return:
        p0Vect - 非侮辱类的条件概率数组
        p1Vect - 侮辱类的条件概率数组
        pAbusive - 文档属于侮辱类的概率
    Modify:
        2018-08-14
    '''
    # 向量文档的长度，本例为6
    numTrainDocs = len(trainMatrix)
    # 向量文档中每篇文档的词条数，本例中为32
    numWords = len(trainMatrix[0])
    # 文档属于侮辱类的概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)
    # 分母denominator初始化为2.0，拉普拉斯平滑
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
        if trainCategory[i] == 1:
            # 每篇向量文档对应相加，如[0,1,0,1]+[1,1,0,1]=[1,2,0,2]
            p1Num += trainMatrix[i]
            # 将每篇文档的各个元素相加，如sum（[1,2,0,1,2]）=6
            p1Denom += sum(trainMatrix[i])
        # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 取对数，防止数据向下溢出
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    # 返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vect, p1Vect, pClass1):
    '''
    Function:
        朴素贝叶斯分类器分类函数
    :param vec2Classify: 待分类的词条数组
    :param p0Vec: 侮辱类的条件概率数组
    :param p1Vec: 非侮辱类的条件概率数组
    :param pClass1: 文档属于侮辱类的概率
    :return:
        0 - 属于非侮辱类
	    1 - 属于侮辱类
	Modify:
        2018-08-14
    '''
    # 对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p1 = sum(vec2Classify * p1Vect) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vect) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    '''
    Function:
        测试朴素贝叶斯分类器
    :return: None
    Modify:
        2018-08-14
    '''
    listOPosts, listClasses = loadDataSet()  # 创建实验样本
    myVocabList = createVocabList(listOPosts)  # 创建词汇表
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))  # 将实验样本向量化
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))  # 训练朴素贝叶斯分类器
    testEntry = ['love', 'my', 'dalmation']  # 测试样本1
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))  # 测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果
    testEntry = ['stupid', 'garbage']  # 测试样本2

    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))  # 测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果


if __name__ == '__main__':
    testingNB()