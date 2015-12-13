#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
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

def random_assign(X_train_array, k):
    min_list = X_train_array.min(axis = 0)
    max_list = X_train_array.max(axis = 0)
    miu = []
    for i in range(k):
        a = []
        for j in range(len(min_list)):
            r = random.random()
            a.append(min_list[j] + (max_list[j] - min_list[j]) * r)
        miu = a
    miu = np.array(miu)
    return miu

def ndim_gaussian(x, miu, var):
    x = np.array(x, dtype = float)
    miu = np.array(miu, dtype = float)
    var = np.array(var, dtype = float)
    n = x.shape[0]
    var_multiply = float(1.0)
    for i in range(var.shape[0]):
        var_multiply *= float(var[i])
    a = np.power(x-miu, 2) / var
    f = 1.0 / (np.sqrt(np.power((2.0 * np.pi), n) * var_multiply)) * np.exp(-0.5* a.sum())
    return f

def predict_train_label(x, mius, vars, priors, label):
#    post_A = priors['A'] * ndim_gaussian(x, mius['A'], vars['A'])
#    post_B = priors['B'] * ndim_gaussian(x, mius['B'], vars['B'])
    if cmp(label, 'A')  == 0 :
        return 'B'
    else:
        return 'A'

def l_func(d, a):
    return 1.0 / (1.0 + np.exp(- (float(a) * float(d))))

def gradient_descent(X_train_dict, mius, vars, priors, a, eps):
    dQ_dmius = {}
    dQ_dmius['A'] = np.zeros(mius['A'].shape[0])
    dQ_dmius['B'] = np.zeros(mius['B'].shape[0])
    dQ_dvars = {}
    dQ_dvars['A'] = np.zeros(vars['A'].shape[0])
    dQ_dvars['B'] = np.zeros(vars['B'].shape[0])
    for label in X_train_dict:
        for x in X_train_dict[label]:
            C = predict_train_label( x, mius, vars, priors, label)
            d = np.log(priors[C] * ndim_gaussian(x, mius[C], vars[C])) - np.log(priors[label] * ndim_gaussian(x, mius[label], vars[label]))
            l_d = l_func(d, a)
            dQ_dmius[C] += a * l_d * (1.0 - l_d) * ((x - mius[C]) / vars[C])
            dQ_dmius[label] += -a * l_d * (1.0 - l_d) * ((x - mius[label]) / vars[label])
            dQ_dvars[C] += a * l_d * (1.0 - l_d) * (0.5 * (np.power((x - mius[C]) / vars[C], 2) - 1.0 / vars[C]))
            dQ_dvars[label] += -a * l_d * (1.0 - l_d) * (0.5 * (np.power((x - mius[label]) / vars[label], 2) - 1.0 / vars[label]))
    new_mius = {}
    new_mius['A'] = mius['A'] - eps * dQ_dmius['A']
    new_mius['B'] = mius['B'] - eps * dQ_dmius['B']
    new_vars = {}
    new_vars['A'] = vars['A'] - eps * dQ_dvars['A']
    new_vars['B'] = vars['B'] - eps * dQ_dvars['B']
    return new_mius, new_vars

def classify(x, mius, vars, priors):
    post_A = priors['A'] * ndim_gaussian(x, mius['A'], vars['A'])
    post_B = priors['B'] * ndim_gaussian(x, mius['B'], vars['B'])
    if post_A >= post_B :
        return 'A'
    else :
        return 'B'

def total_error(X_dict, mius, vars, priors):
    number_of_error = 0
    for label in X_dict:
        for x in X_dict[label]:
            if cmp(classify(x, mius, vars, priors), label) !=0:
                number_of_error += 1
    return number_of_error

def Q_func(X_train_dict, mius, vars, priors):
    Q = float(0)
    for label in X_train_dict:
        for x in X_train_dict[label]:
            C = predict_train_label( x, mius, vars, priors, label)
            d = np.log(priors[C] * ndim_gaussian(x, mius[C], vars[C])) - np.log(priors[label] * ndim_gaussian(x, mius[label], vars[label]))
            l_d = l_func(d, a)
            Q += l_d
    return Q

def pic_plot(Q_array, train_error_array, test_error_array):
    plt.figure(figsize=(18, 9))
    plt.subplot(1, 2, 1)
    plt.title('objective function')
    plt.plot(np.arange(Q_array.shape[0]), Q_array)
    plt.xlabel('iteration')
    plt.ylabel('objective function')

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(train_error_array.shape[0]), train_error_array, 'b-',
             label='train_set')
    plt.plot(np.arange(test_error_array.shape[0]), test_error_array, 'r-',
             label='test_set')
    plt.legend(loc='upper right')
    plt.xlabel('iteration')
    plt.ylabel('classification error rate')
    plt.title('classification error rate')

    plt.show()



if __name__ == '__main__':
    X_train_dict = file2dict('train.txt')
    X_train_dict['A'] = np.array(X_train_dict['A'], dtype= float)
    X_train_dict['B'] = np.array(X_train_dict['B'], dtype= float)
    X_test_dict = file2dict('test.txt')
    X_test_dict['A'] = np.array(X_test_dict['A'], dtype= float)
    X_test_dict['B'] = np.array(X_test_dict['B'], dtype= float)
    mius = {}
    vars = {}
    priors = {}
    total_num = 0
    for label in X_train_dict:
        mius[label] = np.mean(X_train_dict[label], axis = 0)
        vars[label] = np.var(X_train_dict[label], axis = 0)
        total_num += X_train_dict[label].shape[0]
    for label in X_train_dict:
        #priors[label] = float(X_train_dict[label].shape[0]) / float(total_num)
        priors[label] = 0.5
    ''''
    mius['A'] = random_assign(X_train_dict['A'], 1)
    mius['B'] = random_assign(X_train_dict['B'], 1)
    vars['A'] = np.zeros(X_train_dict['A'].shape[1], dtype= float)
    vars['B'] = np.zeros(X_train_dict['B'].shape[1], dtype= float)
    print mius
    for label in vars :
        for i in xrange(X_train_dict[label].shape[0]):
            vars[label] += np.power(X_train_dict[label][i] - mius[label], 2)
    for label in vars:
        vars[label] = vars[label] / X_train_dict[label].shape[0]
    '''
    Q_list = []
    train_error_list = []
    test_error_list = []
    a = 10.0
    eps = 2.0e-3
    print ('Q = %f' %Q_func(X_train_dict, mius, vars, priors) )
    Q_list.append(Q_func(X_train_dict, mius, vars, priors))
    print ('train_error = %f' %(float(total_error(X_train_dict, mius, vars, priors)) / float(X_train_dict['A'].shape[0] + X_train_dict['B'].shape[0])))
    train_error_list.append(float(total_error(X_train_dict, mius, vars, priors)) / float(X_train_dict['A'].shape[0] + X_train_dict['B'].shape[0]))
    print ('test_error = %f' %(float(total_error(X_test_dict, mius, vars, priors)) / float(X_test_dict['A'].shape[0] + X_test_dict['B'].shape[0])))
    test_error_list.append(float(total_error(X_test_dict, mius, vars, priors)) / float(X_test_dict['A'].shape[0] + X_test_dict['B'].shape[0]))
    while True:
        new_mius ,new_vars = gradient_descent(X_train_dict, mius, vars, priors, a, eps)
        m = 0.0
        n = 0.0
        for label in mius :
            p = np.power(mius[label] - new_mius[label], 2)
            m += np.sqrt(p.sum())
            q = np.power(vars[label] - new_vars[label], 2)
            n += np.sqrt(q.sum())
        mius = new_mius
        vars = new_vars
        print ('Q = %f' %Q_func(X_train_dict, mius, vars, priors) )
        Q_list.append(Q_func(X_train_dict, mius, vars, priors))
        print ('train_error = %f' %(float(total_error(X_train_dict, mius, vars, priors)) / float(X_train_dict['A'].shape[0] + X_train_dict['B'].shape[0])))
        train_error_list.append(float(total_error(X_train_dict, mius, vars, priors)) / float(X_train_dict['A'].shape[0] + X_train_dict['B'].shape[0]))
        print ('test_error = %f' %(float(total_error(X_test_dict, mius, vars, priors)) / float(X_test_dict['A'].shape[0] + X_test_dict['B'].shape[0])))
        test_error_list.append(float(total_error(X_test_dict, mius, vars, priors)) / float(X_test_dict['A'].shape[0] + X_test_dict['B'].shape[0]))
#        print float(total_error(X_train_dict, mius, vars, priors)) / float(X_train_dict['A'].shape[0])
#        print float(total_error(X_test_dict, mius, vars, priors)) / float(X_test_dict['A'].shape[0])
        if m < 1e-3 and n <1e-3 :
            break

    Q_array = np.array(Q_list)
    train_error_array = np.array(train_error_list)
    test_error_array = np.array(test_error_list)
    pic_plot(Q_array, train_error_array, test_error_array)