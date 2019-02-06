# -*- coding:utf-8 -*-
# !/usr/bin/python3.6

'''
Function:
    choose 10 text casually to new a folder, and test this program
Modify:
    2018-08-09
'''
import os, operator, random, shutil, KNN
import numpy as np

def extractDataTest():
    # 测试集的Labels
    labels = []
    testFileList = os.listdir('testDigits')
    mTest = len(testFileList)
    for i in range(mTest):
        # 获得文件的名字
        fileNameStr = testFileList[i]
        # 获得不同数字的分类，将列表根据'.'进行切片，得到单独文件名的列表
        classNum = fileNameStr.split('_')[0]
        # 将文件名的后缀剔除，添加到labels列表中，append只用于列表操作
        labels.append(classNum)
    # 去掉重复的元素，进行排序
    single = list(set(labels))
    # single列表按照原来labels的顺序输出
    single.sort(key=labels.index)
    m = len(single)
    # 设置一个随机抽取列表
    stay = []
    for i in range(m):
        # 在0-80之间随机产生10个数，导出到列表
        randList = random.sample(range(0, 80), 10)
        # 将随机数的序号添加到抽取列表，append接受一个对象参数，而extend接受列表参数
        stay.append(randList)
        for j in range(10):
            # 转化为字符串，添加文件的全称（包括后缀）
            combineStr = str(single[i]) + '_' + str(stay[i][j]) + '.txt'
            # print(combineStr)
            # 目标的地址
            newfp = 'C:\\Working\\1.Python\\PyWork\\3.Machining\\1.KNN\\KNN_handWriting\\extractDigits\\' + combineStr
            # 待移动文件的地址
            transStr = 'C:\\Working\\1.Python\\PyWork\\3.Machining\\1.KNN\\KNN_handWriting\\testDigits\\' + combineStr
            try:
                # 移动文件
                shutil.move(transStr, newfp)
            # 首次进行后，随机抽取了10个数字，后面在随机抽取的时候，相同会产生错误
            except FileNotFoundError:
                continue
    print('done!')

def backData():
    testFileList = os.listdir('extractDigits')
    n = len(testFileList)

    for i in range(n):
        # 移动后文件的地址
        newfp = 'C:\\Working\\1.Python\\PyWork\\3.Machining\\1.KNN\\KNN_handWriting\\extractDigits\\' + testFileList[i]
        # 原来的地址
        transStr = 'C:\\Working\\1.Python\\PyWork\\3.Machining\\1.KNN\\KNN_handWriting\\testDigits\\' + testFileList[i]
        shutil.move(newfp, transStr)
        # print(testFileList[i])
    print('done!')


if __name__ == '__main__':
    extractDataTest()
    num = input("Do you want to back the datas?('yes' or 'no')\n")
    if num == 'yes':
        backData()
        print('Datas back to original place!')
    else:
        print('Datas extract successful!')