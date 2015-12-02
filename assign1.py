import numpy as np
from math import exp
from math import sqrt
from math import pow

def classify(x, miu , var):
    pr_A = 1.0 / (sqrt(var['A'][0] * var['A'][1] * var['A'][2]))\
            *exp(-1.0 / 2.0 * (pow((x[0] - miu['A'][0]), 2) / var['A'][0] + pow((x[1] - miu['A'][1]), 2) / var['A'][1] + pow((x[2] - miu['A'][2]), 2) / var['A'][2]))
    pr_B = 1.0 / (sqrt(var['B'][0] * var['B'][1] * var['B'][2]))\
            *exp(-1.0 / 2.0 * (pow(x[0] - miu['B'][0], 2) / var['B'][0] + pow(x[1] - miu['B'][1] , 2) / var['B'][1] + pow(x[2] - miu['B'][2] , 2) / var['B'][2]))
    if(pr_A >= pr_B):
        return 'A'
    else:
        return 'B'

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


X_train_dict = file2dict('train.txt')
X_train_dict['A'] = np.array(X_train_dict['A'])
X_train_dict['B'] = np.array(X_train_dict['B'])
miu = {}
miu['A'] = []
miu['B'] = []
var = {}
var['A'] = []
var['B'] = []
for label in X_train_dict:
    temp_list = X_train_dict[label]
    miu[label] = list(np.mean(X_train_dict[label], axis = 0))
    var[label] = list(np.var(X_train_dict[label], axis = 0))

X_test_dict = file2dict('test.txt')
print X_test_dict
number_of_error = 0
for label in X_test_dict:
    for x in X_test_dict[label]:
        print x
        label_predict = classify(x, miu, var)
        if cmp(label_predict, label) != 0:
            number_of_error += 1

number_of_test = len(X_test_dict['A']) + len(X_test_dict['B'])
error_rate = float(number_of_error)/float(number_of_test)
print number_of_error
print error_rate
print('classification rate : %f' %(1-error_rate))



