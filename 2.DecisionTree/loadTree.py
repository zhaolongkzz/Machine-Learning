# -*- coding:utf-8 -*-
# !/usr/bin/python3.6

import pickle, json

def grabTree(filename):
    '''
    Function:
        读取决策树
    :param filename: 决策树的存储文件名
    :return:
        pickle.load(fr) - 决策树字典
    Modify:
        2018-08-12
    '''
    # 原因如storeTree所示
    # fr = open(filename, 'rb')
    # return pickle.load(fr)
    fr = open(filename, 'rb')
    return json.load(fr)


if __name__ == '__main__':
    myTree = grabTree('classifierStorage.txt')
    print(myTree)