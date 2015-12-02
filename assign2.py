#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from math import sqrt
from math import exp
from math import pi
from math  import pow
import random

def file2dict(filename):
    file = open(filename)
    array_of_lines = file.readlines()
    X_dict = {}
    X_dict['A'] = []
    X_dict['B'] = []
    for line in array_of_lines :
        line = line.strip('\n')
        temp_list = line.split('(')
        if 'A: ' in temp_list:
            temp_list[1] = temp_list[1].strip(')')
            temp_list[1] = temp_list[1].split(' ')
            for i in range(len(temp_list[1])):
                temp_list[1][i] = float(temp_list[1][i])
            X_dict['A'].append(temp_list[1])
        if 'B: ' in temp_list:
            temp_list[1] = temp_list[1].strip(')')
            temp_list[1] = temp_list[1].split(' ')
            for i in range(len(temp_list[1])):
                temp_list[1][i] = float(temp_list[1][i])
            X_dict['B'].append(temp_list[1])
    file.close()
    return X_dict

def distance(vecA, vecB):##vecA 和 vecB都是np.array
    D = sqrt(np.sum(np.power(vecA - vecB, 2)))
    return D

def random_assign(X_train_list, k):
    min_list = X_train_list.min(axis = 0)
    max_list = X_train_list.max(axis = 0)
    miu = []
    for i in range(k):
        a = []
        for j in range(len(min_list)):
            k = random.random()
            a.append(min_list[j] + (max_list[j] - min_list[j])*k)
        miu.append(a)
    return miu

def k_means(X_train_list, k):
    labels = [0]*X_train_list.shape[0]
    distances = ['inf']*X_train_list.shape[0]
    mius = random_assign(X_train_list, k)
    mius = np.array(mius)
    new_mius = np.zeros(mius.shape)
    while True:
        for i in range(X_train_list.shape[0]):
            for j in range(mius.shape[0]):
                d = distance(X_train_list[i], mius[j])
                if d < distances[i]:
                    distances[i] = d
                    labels[i] = j
        for j in range(new_mius.shape[0]):
            indexs = [i for i,a in enumerate(labels) if a== j]
            s = np.zeros(X_train_list.shape[1])
            for i in indexs:
                s += X_train_list[i]
            new_mius[j] = s/len(indexs)
        a = np.power(mius - new_mius, 2)
        b = np.sqrt(a.sum(axis = 1))
        mius = new_mius
        if b.sum()< 1e-5 :
            break
    return mius, labels

def ndim_gaussian(x, miu, var):
    n = x.shape[0]
    var_multiply = 1.0
    for i in range(var.shape[0]):
        var_multiply *= var[i]
    a = np.power(x-miu, 2) / var
    f = 1.0 / (sqrt(pow((2.0 * pi), n) * var_multiply)) * exp(-1.0 / 2.0 * a.sum())
    return f

def GMM_EM(X_train_list, k, phis_in, mius_in, vars_in):
    w = np.zeros((X_train_list.shape[0], k))
    w_new = np.zeros((X_train_list.shape[0], k))
    phis = phis_in
    mius = mius_in
    vars = vars_in
    while True:
        for i in range(w.shape[0]):
            m = 0.0
            for j in range(w.shape[1]):
                m += ndim_gaussian(X_train_list[i], mius[j], vars[j]) * phis[j]
            for j in range(w.shape[1]):
                w[i][j] = (ndim_gaussian(X_train_list[i], mius[j], vars[j]) * phis[j])/m
        for j in range(w.shape[1]):
            s1 = 0.0
            s2 = 0.0
            s3 = 0.0
            for i in range(w.shape[0]):
                s1 += w[i][j]
                s2 += w[i][j] * X_train_list[i]
                s3 += w[i][j] * (X_train_list[i] - mius[j]) * (X_train_list[i] - mius[j])
            phis[j] = s1 / w.shape[0]
            mius[j] = s2 / s1
            vars[j] = s3 / s1
        for i in range(w.shape[0]):
            m = 0.0
            for j in range(w.shape[1]):
                m += ndim_gaussian(X_train_list[i], mius[j], vars[j]) * phis[j]
            for j in range(w.shape[1]):
                w_new[i][j] = (ndim_gaussian(X_train_list[i], mius[j], vars[j]) * phis[j])/m
        a = np.power(w - w_new, 2)
        b = np.sqrt(a.sum(axis = 1))
        if b.sum()< 1e-5 :
            break
    return phis, mius, vars



X_train_dict = file2dict('train.txt')
X_train_list = X_train_dict['A'] + X_train_dict['B']
X_train_list = np.array(X_train_list)
k = 2
mius, labels = k_means(X_train_list, k)
vars = np.zeros(mius.shape, dtype = np.float64)
phis = np.zeros(mius.shape[0], dtype = np.float64)
numbers = np.zeros(mius.shape[0])
for j in range(k):
    numbers[j] = len([i for i,a in enumerate(labels) if a== j])
for i in range(len(labels)):
    vars[labels[i]] += np.power(X_train_list[i] - mius[labels[i]], 2)
for i in range(vars.shape[0]):
    vars[i] = vars[i] / float(numbers[i])
phis = numbers/float(X_train_list.shape[0])
phis, mius, vars = GMM_EM(X_train_list, k, phis, mius, vars)
print phis
print mius
print vars

