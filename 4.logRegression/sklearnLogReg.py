# -*- coding:utf-8 -*-
# !/usr/bin/python3.6

from sklearn.linear_model import LogisticRegression
import time

def colicSklearn():
    '''
    Functions:
        使用Sklearn构建Logistic回归分类器
    :return: None
    Modify:
        2018-09-12
    '''
    frTrain = open('horseColicTraining.txt')  # 打开训练集
    frTest = open('horseColicTest.txt')  # 打开测试集
    trainingSet = []; trainingLabels = []
    testSet = []; testLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))
    classifier = LogisticRegression(solver='liblinear', max_iter=10).fit(trainingSet, trainingLabels)
    # classifier = LogisticRegression(solver='sag', max_iter=5000).fit(trainingSet, trainingLabels)   # 随机平均梯度下降算法
    test_accurcy = classifier.score(testSet, testLabels) * 100
    print('正确率:%f%%' % test_accurcy)


if __name__ == '__main__':
    start = time.clock()
    colicSklearn()
    end = time.clock()
    print('总耗时为：' + str(end - start) + 's')