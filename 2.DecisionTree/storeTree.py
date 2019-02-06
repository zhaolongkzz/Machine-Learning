# -*- coding:utf-8 -*-
# !/usr/bin/python3.6

import pickle, json

def storeTree(inputTree, filename):
    '''
    Function:
        存储决策树
    :param inputTree: 已经生成的决策树
    :param filename: 决策树的存储文件名
    :return: None
    Modify:
        2018-08-12
    '''
    # fw = open(dilename, 'w')
    # pickle.dump(inputTree, fw)
    # fw.close()
    # 或者以下方式，但是保存中文会乱码，故使用json来保存
    # with open(filename, 'wb') as fw:
    #     pickle.dump(inputTree, fw)

    with open(filename, 'w', encoding='utf-8') as fw:
        json.dump(inputTree, fw, ensure_ascii=False)


if __name__ == '__main__':
    myTree = {'有自己的房子': {0: {'有工作': {0: 'no', 1: {'信贷情况': {0: 'no', 1: 'yes', 2: 'yes'}}}}, 1: 'yes'}}
    storeTree(myTree, 'classifierStorage.txt')