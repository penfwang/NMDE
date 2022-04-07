from __future__ import division
import bisect
import math
import random
from itertools import chain
from operator import attrgetter, itemgetter
from collections import defaultdict
import numpy as np
import itertools
from minepy import MINE
from deap import tools, base
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
toolbox = base.Toolbox()



def fit_train(x1, train_data):
    x = np.zeros((1,len(x1)))
    for ii in range(len(x1)):
        x[0,ii] = x1[ii]
    x = random.choice(x)
    x = 1 * (x >= 0.6)
    if np.count_nonzero(x) == 0:
        f1 = 1###error_rate
        f2 = 1
    else:
     x = "".join(map(str, x))  # transfer the array form to string in order to find the position of 1
     value_position = np.array(list(find_all(x, '1'))) + 1  # cause the label in the first column in training data
     value_position = np.insert(value_position, 0, 0)  # insert the column of label
     tr = train_data[:, value_position]
     clf = KNeighborsClassifier(n_neighbors = 5)
     scores = cross_val_score(clf, tr[:,1:],tr[:,0], cv = 10)
     f1 = np.mean(1 - scores)
     f2 = (len(value_position)-1)/(train_data.shape[1] - 1)
     # f2 = len(value_position) - 1
    return f1, f2


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)

def findindex(org, x):
    result = []
    for k,v in enumerate(org): #k和v分别表示org中的下标和该下标对应的元素
        if v == x:
            result.append(k)
    return result


def obtain_r(z, x_train, mic_value):
    tt_01 = 1 * (np.array(z) >= 0.6)
    tt_01 = "".join(map(str, tt_01))  ######## the '0101001' of the current individual
    z_index = np.array(list(find_all(tt_01, '1')))  ##### the selected features of the current individual
    r_cf = mic_value[z_index]  ########## the related mic of the features with label
    average_r_cf = np.mean(r_cf)
    aa = list(itertools.combinations(z_index, 2))### the unique combinations between pair of features
    mine = MINE(alpha=0.6, c=15)
    mic_ff = []
    for i_in in range(len(aa)):
           mine.compute_score(x_train[:, aa[i_in][0] + 1], x_train[:, aa[i_in][1] + 1])
           mic_ff.append(mine.mic())
    mic_ff = np.array(mic_ff)
    average_r_ff = np.mean(mic_ff)
    k =len(tt_01)
    m_value = (k * average_r_cf) / math.sqrt(k + k * (k - 1) * average_r_ff)
    return m_value

def ran(number):
    i = 0
    y = np.zeros((number))
    while i< number:
        y[i]=random.random()
        i = i+1
    return y


def add_delete(temp,p_add,p_delete,dim):
    y_add = random.random()
    y_delete = random.random()
    inter = random.randint(0,dim-1)
    if y_add < p_add[inter]:
            temp[inter] = 1
    if y_delete < p_delete[inter]:
            # temp[t2] = random.uniform(0, 0.6)
            temp[inter] = 0
    return temp

def get_whole_01(individuals):
    all_index = []
    individuals_array = np.array(individuals)  ####
    for i0 in range(individuals_array.shape[0]):
        x1 = 1 * (individuals_array[i0, :] >= 0.6)
        x1 = "".join(map(str, x1))  # transfer the array form to string in order to find the position of 1
        all_index.append(x1)  ##store all individuals who have changed to 0 or 1
    return all_index

def mutDE(a, b, c):###mutation:DE/rand/1
    f = 0.8
    y = toolbox.clone(a)
    for i in range(len(y)):
        y[i] = a[i] + f*(b[i]-c[i])
        if y[i] > 1:
            y[i] = 1
        if y[i] < 0:
            y[i] = 0
    return y


def DE_mutation(temp,pop_non):
    if len(pop_non) == 0:
        temp_new = temp
    elif len(pop_non) == 1:
        index = [0,0]
        b = pop_non[index[0]]
        c = pop_non[index[1]]
        temp_new = mutDE(temp, b, c)
    else:
      index = random.sample(range(0,len(pop_non)),2)
      b = pop_non[index[0]]
      c = pop_non[index[1]]
      temp_new = mutDE(temp, b, c)
    return temp_new


def produce_diverse_individuals(individuals,pop_non,pop_surrogate):
    if len(individuals) == 0:
        return
    matrix_01 = np.zeros((len(pop_non),len(pop_non[0])))
    EXA_array = np.array(pop_non)
    for i0 in range(EXA_array.shape[0]):
        x = 1 * (EXA_array[i0, :] >= 0.6)
        matrix_01[i0,:] = x
    p_add = matrix_01.mean(axis=0) + 0.2
    p_delete = 1 - matrix_01.mean(axis=0) + 0.2
    #####################################################get the probability to add or delete one feature
    all_index = get_whole_01(individuals)
    unique_all_Index = set(all_index)
    unique_all_Index = list(unique_all_Index)
    index_unique = [0.0] * len(unique_all_Index)
    index_duplication = []
    for i1 in range(len(unique_all_Index)):
        index_of_objectives = findindex(all_index, unique_all_Index[i1])
        index_unique[i1] = index_of_objectives[0]
        index_duplication.extend([index_of_objectives[1:]])
    print(index_unique)
    print(index_duplication)
    exit()
    ##################################################################get the index of unique and duplicated solutions
    for i1 in index_duplication:
        if i1 != []:#####some feature combinations no duplications
           for i2 in range(len(i1)):###some feature combinations may have more than one duplications
             ii = 1
             while ii < 100:
                 individuals[i1[i2]] = add_delete(individuals[i1[i2]],p_add,p_delete,len(pop_non[0]))###hang
                 individuals[i1[i2]] = DE_mutation(individuals[i1[i2]], pop_non)###from pop_non choose solution to mutate
                 new_one1 = np.array(individuals[i1[i2]])
                 new_one_011 = 1 * (new_one1 >= 0.6)
                 new_one_011 = "".join(map(str, new_one_011))
                 temp = findindex(unique_all_Index, new_one_011)
                 del all_index,unique_all_Index
                 all_index = get_whole_01(individuals)
                 unique_all_Index = set(all_index)
                 unique_all_Index = list(unique_all_Index)
                 if len(temp) == 0:
                     break
                 del temp
                 ii = ii + 1
 #####################################################思想是继续check是否已经存在，已经存在就继续直到生成新的unique解
    return individuals




def improved_evaluation(individuals,pop_unique, fit_num,x_train):
    pop_unique_fit = np.array([ind.fitness.values for ind in pop_unique])
    if len(individuals) == 0:
        return
    fit_new = np.zeros((len(individuals),2))
    all_index_pop_unique = get_whole_01(pop_unique)
    all_index_children = get_whole_01(individuals)#####offspring's 01 combinations
    # print(all_index_children)
    unique_all_Index_children = set(all_index_children)
    unique_all_Index_children = list(unique_all_Index_children)
    index_unique_children = [0.0] * len(unique_all_Index_children)
    index_duplication_children = []
    for i1 in range(len(unique_all_Index_children)):
        index_of_objectives_children = findindex(all_index_children, unique_all_Index_children[i1])
        index_unique_children[i1] = index_of_objectives_children[0]#######the first store unique set
        index_duplication_children.extend([index_of_objectives_children[1:]])######the remaining store duplicated set
    # print(index_unique_children)
    # print(index_duplication_children)
    # exit()
    for i2 in range(len(index_unique_children)):
        index_unique_children_pop_unique = findindex(all_index_pop_unique, all_index_children[index_unique_children[i2]])
        # print('index', all_index_children[index_unique_children[i2]], index_unique_children_pop_unique)
        # exit()
        if len(index_unique_children_pop_unique) == 0:#####it means there's a new solution never come up
            fit_new[index_unique_children[i2]] = fit_train(individuals[index_unique_children[i2]], x_train)
            fit_num = fit_num + 1
        else:#####it is a duplicated solution in unique set
            # print('len(pop_unique_fit)',len(pop_unique_fit))
            # print('len(all_index_pop_unique)',len(all_index_pop_unique))
            fit_new[index_unique_children[i2]] = pop_unique_fit[index_unique_children_pop_unique[0]]
        if len(index_duplication_children[i2]) != 0:##########consider the solutions in the duplicated set
            for i3 in range(len(index_duplication_children[i2])):
                fit_new[index_duplication_children[i2][i3]] = fit_new[index_unique_children[i2]]
    # print(fit_new)
    # exit()
    return fit_new, fit_num










def surrogate(before, after, surrogate):
    print(before)
    print(after)
    exit()

    return surrogate


def produce_diverse_individuals_surrogate(individuals,pop_non,pop_surrogate):
    if len(individuals) == 0:
        return
    matrix_01 = np.zeros((len(pop_non),len(pop_non[0])))
    EXA_array = np.array(pop_non)
    for i0 in range(EXA_array.shape[0]):
        x = 1 * (EXA_array[i0, :] >= 0.6)
        matrix_01[i0,:] = x
    p_add = matrix_01.mean(axis = 0) + 0.2
    p_delete = 1- matrix_01.mean(axis = 0) + 0.2
    #####################################################get the probability to add or delete one feature
    all_index = get_whole_01(individuals)
    unique_all_Index = set(all_index)
    unique_all_Index = list(unique_all_Index)
    # print(all_index)
    # print(unique_all_Index)
    index_unique = [0.0] * len(unique_all_Index)
    index_duplication = []
    for i1 in range(len(unique_all_Index)):
        index_of_objectives = findindex(all_index, unique_all_Index[i1])
        index_unique[i1] = index_of_objectives[0]
        index_duplication.extend([index_of_objectives[1:]])
    ##################################################################get the index of unique and deplicated solutions
    for i1 in index_duplication:
        if i1 != []:#####some feature combinations no duplications
           for i2 in range(len(i1)):###some feature combinations may have more than one duplications
             ii = 1
             while ii < 100:
                 before = toolbox.clone(individuals[i1[i2]])
                 # before_fitness = np.array([ind.fitness.values for ind in before])
                 new = add_delete(individuals[i1[i2]],p_add,p_delete,len(pop_non[0]))
                 print(before)
                 print(new)
                 exit()
                 new_one1 = np.array(individuals[i1[i2]])
                 new_one_011 = 1 * (new_one1 >= 0.6)
                 new_one_011 = "".join(map(str, new_one_011))
                 temp = findindex(unique_all_Index, new_one_011)
                 del all_index,unique_all_Index
                 all_index = get_whole_01(individuals)
                 unique_all_Index = set(all_index)
                 unique_all_Index = list(unique_all_Index)
                 if len(temp)==0:
                     break
                 ii = ii + 1
                 print(before)
                 exit()
             individuals[i1[i2]] = surrogate(before, individuals[i1[i2]],pop_surrogate)
    return individuals