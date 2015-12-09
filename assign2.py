#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt, exp, pi, pow
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

def random_assign(X_train_array, k):
    min_list = X_train_array.min(axis = 0)
    max_list = X_train_array.max(axis = 0)
    miu = []
    for i in range(k):
        a = []
        for j in range(len(min_list)):
            r = random.random()
            a.append(min_list[j] + (max_list[j] - min_list[j]) * r)
        miu.append(a)
    return miu

def k_means(X_train_array, k, eps):
    labels = [0]*X_train_array.shape[0]
    distances = ['inf']*X_train_array.shape[0]
    mius = random_assign(X_train_array, k)
    mius = np.array(mius)
    new_mius = np.zeros(mius.shape)
    while True:
        for i in range(X_train_array.shape[0]):
            for j in range(mius.shape[0]):
                d = distance(X_train_array[i], mius[j])
                if d < distances[i]:
                    distances[i] = d
                    labels[i] = j
        for j in range(new_mius.shape[0]):
            indexs = [i for i,a in enumerate(labels) if a== j]
            s = X_train_array[indexs].sum()
            if len(indexs) == 0:
                new_mius[j] = np.array([float('inf')]*new_mius[j].shape[0])
            else:
                new_mius[j] = s / len(indexs)
        for j in range(new_mius.shape[0]):
            if float('inf') in list(new_mius[j]):
                mius = new_mius
                return mius, labels
        a = np.power(mius - new_mius, 2)
        b = np.sqrt(a.sum(axis = 1))

        mius = new_mius
        if b.sum()< eps :
            break
    return mius, labels

def ndim_gaussian(x, miu, var):
    if float('inf') in miu or float('inf') in var :
        return 0
    n = x.shape[0]
    var_multiply = 1.0
    for i in range(var.shape[0]):
        var_multiply *= var[i]
    a = np.power(x-miu, 2) / var
    f = 1.0 / (sqrt(pow((2.0 * pi), n) * var_multiply)) * exp(-1.0 / 2.0 * a.sum())
    return f

def GMM_EM(X_train_array, k, phis_in, mius_in, vars_in, eps):
    w = np.zeros((X_train_array.shape[0], k))
    w_new = np.zeros((X_train_array.shape[0], k))
    phis = phis_in
    mius = mius_in
    vars = vars_in
    while True:
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                w[i][j] = ndim_gaussian(X_train_array[i], mius[j], vars[j]) * phis[j]
            w[i] = w[i] / w[i].sum()
        for j in range(w.shape[1]):
            s1 = 0.0
            s2 = np.zeros(mius[j].shape)
            s3 = np.zeros(vars[j].shape)
            for i in range(w.shape[0]):
                s1 += w[i][j]
                s2 += w[i][j] * X_train_array[i]
                if float('inf') in mius[j] :
                    s3 += np.zeros(vars[j].shape)
                else:
                    s3 += w[i][j] * (X_train_array[i] - mius[j]) * (X_train_array[i] - mius[j])
            if float('inf') in mius[j]:
                phis[j] = phis[j]
                mius[j] = mius[j]
                vars[j] = vars[j]
            else:
                phis[j] = s1 / w.shape[0]
                mius[j] = s2 / s1
                vars[j] = s3 / s1
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                w_new[i][j] = ndim_gaussian(X_train_array[i], mius[j], vars[j]) * phis[j]
            w_new[i] = w_new[i] / w_new[i].sum()
        a = np.power(w - w_new, 2)
        b = np.sqrt(a.sum(axis = 1))
        if b.sum()< eps :
            break
    return phis, mius, vars

def classify(x, mius, vars, phis):
    post = {}
    for label in phis:
        k = phis[label].shape[0]
        post[label] = 0.0
        for i in range(k):
            post[label] += phis[label][i] * ndim_gaussian(x, mius[label][i], vars[label][i])
    post_2 = post.items()
    post_2.sort(key = lambda one_post : one_post[1], reverse= True)
    return post_2[0][0]


if __name__ == '__main__':
    X_train_dict = file2dict('train.txt')
    X_train_dict['A'] = np.array(X_train_dict['A'])
    X_train_dict['B'] = np.array(X_train_dict['B'])
    X_test_dict = file2dict('test.txt')
    X_test_dict['A'] = np.array(X_test_dict['A'])
    X_test_dict['B'] = np.array(X_test_dict['B'])
    k = 8
    error_rate = 1.0
    while error_rate > 0.19:
        mius = {}
        vars = {}
        phis = {}
        for label in X_train_dict:
            mius[label], classes = k_means(X_train_dict[label], k, 1.0e-5)
            vars[label] = np.zeros(mius[label].shape, dtype = np.float64)
            phis[label] = np.zeros(k, dtype = np.float64)
            numbers = np.zeros(k)
            for j in range(k):
                numbers[j] = len([i for i,a in enumerate(classes) if a== j])
            for i in range(len(classes)):
                vars[label][classes[i]] += np.power(X_train_dict[label][i] - mius[label][classes[i]], 2)
            for i in range(vars[label].shape[0]):
                if numbers[i] == 0:
                    vars[label][i] = np.array([float('inf')]*vars[label][i].shape[0])
                else:
                    vars[label][i] = vars[label][i] / float(numbers[i])
            phis[label] = numbers/float(X_train_dict[label].shape[0])
            phis[label], mius[label], vars[label] = GMM_EM(X_train_dict[label], k, phis[label], mius[label], vars[label], 1.0e-2)

        number_of_error = 0
        for label in X_test_dict:
            for x in X_test_dict[label]:
                label_predict = classify(x, mius, vars, phis)
                if cmp(label_predict, label) != 0:
                    number_of_error += 1

        number_of_test = len(X_test_dict['A']) + len(X_test_dict['B'])
        error_rate = float(number_of_error)/float(number_of_test)
        print mius
        print vars
        print phis
        print ('number of classification error: %d'%(number_of_error))
        print ('classification error rate: %f'%(error_rate))
        print('classification rate : %f' %(1-error_rate))