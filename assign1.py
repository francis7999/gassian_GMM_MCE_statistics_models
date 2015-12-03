#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from math import exp, sqrt, pow, pi

def ndim_gaussian(x, miu, var):##x, miu, var为同维array
    n = x.shape[0]
    var_multiply = 1.0
    for i in range(var.shape[0]):
        var_multiply *= var[i]
    a = np.power(x-miu, 2) / var
    f = 1.0 / (sqrt(pow((2.0 * pi), n) * var_multiply)) * exp(-1.0 / 2.0 * a.sum())
    return f

def classify(x, miu , var, prior):
    post = {}
    for label in miu:
        post[label] = prior[label] * ndim_gaussian(x, miu[label], var[label])
    post_2 = post.items()
    post_2.sort(key = lambda one_post : one_post[1], reverse= True)
    return post_2[0][0]

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

if __name__ == '__main__':
    X_train_dict = file2dict('train.txt')
    X_train_dict['A'] = np.array(X_train_dict['A'])
    X_train_dict['B'] = np.array(X_train_dict['B'])
    miu = {}
    var = {}
    prior = {}
    prior['A'] = float(X_train_dict['A'].shape[0]) / float(X_train_dict['A'].shape[0] + X_train_dict['B'].shape[0])
    prior['B'] = float(X_train_dict['B'].shape[0]) / float(X_train_dict['A'].shape[0] + X_train_dict['B'].shape[0])
    for label in X_train_dict:
        miu[label] = np.mean(X_train_dict[label], axis = 0)
        var[label] = np.var(X_train_dict[label], axis = 0)

    X_test_dict = file2dict('test.txt')
    X_test_dict['A'] = np.array(X_test_dict['A'])
    X_test_dict['B'] = np.array(X_test_dict['B'])
    number_of_error = 0
    for label in X_test_dict:
        for x in X_test_dict[label]:
            label_predict = classify(x, miu, var, prior)
            if cmp(label_predict, label) != 0:
                number_of_error += 1

    number_of_test = len(X_test_dict['A']) + len(X_test_dict['B'])
    error_rate = float(number_of_error)/float(number_of_test)
    print number_of_error
    print error_rate
    print('classification rate : %f' %(1-error_rate))



