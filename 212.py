# -*- coding:utf-8 -*-
# !/usr/bin/python3

'''
test the part program when you do not know!
'''

from math import log, sqrt
import operator, re, random
import numpy as np
import tensorflow as tf
import sys, scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd

# a = 333
# b = 333
# print(a is b)


# value = [3, 42, 43, 5]
# for item in reversed(value):
#     print(item)


import numpy as np
import matplotlib.pyplot as plt
# x = np.linspace(0, 2, 1000)
# y = x ** 2
# plt.plot(x, y)
# plt.fill_between(x, y, where=(y > 0), color='red', alpha=0.5)
# # plt.show()
#
# N = 1000
# points = [[xy[0] * 2, xy[1] * 4] for xy in np.random.rand(N, 2)]
# plt.scatter([x[0] for x in points], [x[1] for x in points], s=5, c=np.random.rand(N), alpha=0.5)
# plt.show()
#
# count = 0
# for xy in points:
#     if xy[1] < xy[0] ** 2:
#         count += 1
# print((count / N) * (2 * 4))
#
# # print(scipy.integrate.quad(lambda x: x ** 2, 0, 2)[0])
#
# print(lambda x: x ** 2, 0, 2)


# Set = []
# data = open('a.txt')
# for line in data.readlines():
#     currLine = line.split(' ')
#     for i in range(len(currLine) - 1):
#         print(currLine)


# Book = namedtuple('Book', ['id', 'title', 'authors'])
# Book.__doc__ += ': Hardcover book in active collection'
# Book.id.__doc__ = '13-digit ISBN'
# Book.title.__doc__ = 'Title of first printing'
# Book.authors.__doc__ = 'List of authors sorted by last name'


m = nn.Linear(2, 3, bias=False)
n = nn.Linear(2, 3)
print(m)
input = torch.autograd.Variable(torch.randn(2, 2))
print(input)
output = m(input)
output1 = n(input)
print(output)
print(output1)
print(output.size())
