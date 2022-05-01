from __future__ import division
import random
import numpy as np
from deap import base
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
     scores = cross_val_score(clf, tr[:,1:],tr[:,0], cv = 5)
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
    for k,v in enumerate(org): 
        if v == x:
            result.append(k)
    return result


def get_whole_01(individuals):
    all_index = []
    individuals_array = np.array(individuals)  ####
    for i0 in range(individuals_array.shape[0]):
        x1 = 1 * (individuals_array[i0, :] >= 0.6)
        x1 = "".join(map(str, x1))  # transfer the array form to string in order to find the position of 1
        all_index.append(x1)  ##store all individuals who have changed to 0 or 1
    return all_index





def improved_evaluation(individuals,pop_unique, fit_num,x_train):
    pop_unique_fit = np.array([ind.fitness.values for ind in pop_unique])
    if len(individuals) == 0:
        return
    fit_new = np.zeros((len(individuals),2))
    all_index_pop_unique = get_whole_01(pop_unique)
    all_index_children = get_whole_01(individuals)#####offspring's 01 combinations
    unique_all_Index_children = set(all_index_children)
    unique_all_Index_children = list(unique_all_Index_children)
    index_unique_children = [0.0] * len(unique_all_Index_children)
    index_duplication_children = []
    for i1 in range(len(unique_all_Index_children)):
        index_of_objectives_children = findindex(all_index_children, unique_all_Index_children[i1])
        index_unique_children[i1] = index_of_objectives_children[0]#######the first store unique set
        index_duplication_children.extend([index_of_objectives_children[1:]])######the remaining store duplicated set
    for i2 in range(len(index_unique_children)):
        index_unique_children_pop_unique = findindex(all_index_pop_unique, all_index_children[index_unique_children[i2]])
        if len(index_unique_children_pop_unique) == 0:#####it means there's a new solution never come up
            fit_new[index_unique_children[i2]] = fit_train(individuals[index_unique_children[i2]], x_train)
            fit_num = fit_num + 1
        else:#####it is a duplicated solution in unique set
            fit_new[index_unique_children[i2]] = pop_unique_fit[index_unique_children_pop_unique[0]]
        if len(index_duplication_children[i2]) != 0:##########consider the solutions in the duplicated set
            for i3 in range(len(index_duplication_children[i2])):
                fit_new[index_duplication_children[i2][i3]] = fit_new[index_unique_children[i2]]
    return fit_new, fit_num
