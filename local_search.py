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
import geatpy as ea


def fit_train(x1, train_data):
    x = np.zeros((1,len(x1)))
    for ii in range(len(x1)):
        x[0,ii] = x1[ii]
    x= random.choice(x)
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



def get_whole_01(individuals):
    all_index = []
    individuals_array = np.array(individuals)  ####
    for i0 in range(individuals_array.shape[0]):
        x1 = 1 * (individuals_array[i0, :] >= 0.6)
        x1 = "".join(map(str, x1))  # transfer the array form to string in order to find the position of 1
        all_index.append(x1)  ##store all individuals who have changed to 0 or 1
    return all_index

def mutDE(y, a, b, c):###mutation:DE/rand/1
    f = 0.8
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
        y = toolbox.clone(temp)
        temp_new = mutDE(y, temp, b, c)
    else:
      index = random.sample(range(0,len(pop_non)),2)
      b = pop_non[index[0]]
      c = pop_non[index[1]]
      y = toolbox.clone(temp)
      temp_new = mutDE(y, temp, b, c)
    return temp_new



def calculate_fruquency(pop):
    matrix_01 = np.zeros((len(pop),len(pop[0])))
    EXA_array = np.array(pop)
    for i0 in range(EXA_array.shape[0]):
        x = 1 * (EXA_array[i0, :] >= 0.6)
        matrix_01[i0,:] = x
    # print('matrix_01',matrix_01)
    f = matrix_01.mean(axis = 0)
    # print(matrix_01)
    return f


def decision_average(ins):
    matrix = np.zeros((len(ins), len(ins[0])))
    for i in range(len(ins)):
        matrix[i,:] = ins[i]
    new = matrix.mean(axis = 0)
    return new

def first_nondominated(PF):
    [levels1, criLevel1] = ea.indicator.ndsortDED(PF, 1)
    x1 = 1 * (levels1 == 1.0)
    x1 = "".join(map(str, x1))
    index_non = np.array(list(find_all(x1, '1')))
    return index_non


def Gaussian_produce_solution(ins,u,pop_unique,pop_unique_fit,fit_num,x_train,feature_possible):
    ###ins stores the solutions who have the same classification performance
    matrix = np.zeros((len(ins), len(ins[0])))
    matrix1 = np.zeros((len(ins), len(ins[0])))
    delete_solution = []
    add_solution = []
    # for i in range(len(ins)):
    #     matrix[i,:] = ins[i]
    #     matrix1[i,:] = (matrix[i,:] - u)**2###part of the processes to calculate sigma
    # sigma = matrix1.mean(axis=0)
    ####################################################################whether u has shown before
    fit = []
    u1 = 1 * (u >= 0.6)
    # print(u1)
    all_index_children = "".join(map(str, u1))
    value_posi = np.array(list(find_all(all_index_children, '1'))) + 1
    # print('selected positions from u',value_posi)
    if len(value_posi) == 0:
       u3 = 1 * (feature_possible > 0.1)
       u = u3
    all_index_pop_unique = get_whole_01(pop_unique)
    # for i in range(1):########check u whether in pop_unique, later maybe there will have more than one solution
    index = findindex(all_index_pop_unique, all_index_children)
    if len(index) == 0:#####It means there's a new solution which needs to calculate its fitness values
        fit.append(fit_train(u, x_train))##calculate new solution's fitness
        fit_num = fit_num + 1
        fit.extend([ind.fitness.values for ind in ins])##insert the fitness of previous solutions with the same performance
        fit = np.array(fit)
        index_n = first_nondominated(fit)##sort based the non-dominated concept
        u2 = toolbox.clone(ins[0])
        for i in range(len(u2)):
                u2[i] = u[i]
        u2.fitness.values = fit[0]
        pop_unique.append(u2)################################################################add
        if 0 in index_n:##new solution is better. u is the solution with the same accuracy, but may have the same size.
                delete_solution.extend(ins)
                add_solution.append(u2)
        # return delete_solution,add_solution,fit_num,pop_unique
    else:#####it is a duplicated solution in unique set
                 delete_solution = []
                 add_solution = []
    return delete_solution,add_solution,fit_num,pop_unique




def produce_new_from_frequency(pop,feature_possible):
    new_solution = toolbox.clone(pop[0])
    for j in range(len(new_solution)):
        new_solution[j] = random.uniform(0.0, 0.6)
    index = np.argwhere(feature_possible > 1/len(pop))
    for i in index:
        i = random.choice(i)
        new_solution[i] = random.uniform(0.6, 1)
    return new_solution


def combine_all_solutions(pop, feature_possible):
    new_solution = toolbox.clone(pop[0])
    for j in range(len(new_solution)):
        new_solution[j] = random.uniform(0.0, 0.6)
    index = np.argwhere(feature_possible > 0)
    for i in index:
        i = random.choice(i)
        new_solution[i] = random.uniform(0.6, 1)
    return new_solution



def check_whether_it_is_shown(u2,unique_set):
    u = np.array(u2)
    u1 = 1 * (u >= 0.6)
    all_index_children = "".join(map(str, u1))
    index = findindex(unique_set, all_index_children)
    return index



def p_local_search(pop,pop_unique,f_eva,x_train,ee):
    if len(pop) == 0:
        return pop, f_eva
    EXA_fit = np.array([ind.fitness.values for ind in pop])
    EXA_0_list = list(EXA_fit[:, 0])#####error rate
    EXA_1_list = list(EXA_fit[:, 1])#####subset size
    single1_index = set(EXA_1_list)
    single1_index = list(single1_index)
    all_index_pop_unique = get_whole_01(pop_unique)
    if len(single1_index) == len(pop):#### this means the solutions have different sizes that means they are all unique.
        return pop, f_eva, pop_unique
    else:###############according to the size index to find the error within the range ee,
      for i in range(len(single1_index)):
        index_of_size = findindex(EXA_fit[:,1], single1_index[i])###the index of some solutions with the same size(size changes)
        if len(index_of_size) > 1:###multiple solutions have the same size, if abs(PF_new[temp,0]- temp1[0]) <= ee:
            temp0 = [EXA_0_list[m] for m in index_of_size]
            index_error = np.argwhere(abs(temp0 - min(temp0)) <= ee)
            list1 = []
            if len(index_error) > 1:
                for ii in index_error:
                      iii = random.choice(ii)
                      list1.append(iii)
                list2 = [index_of_size[t] for t in list1]####size same, and error in a range same
                ins = [pop[t] for t in list2]
                  #####################################################feature frequency
                feature_possible = calculate_fruquency(ins)
                if max(feature_possible) > 1/len(ins): ###situation 1 or 3
                  new_solution = produce_new_from_frequency(ins,feature_possible)
                  ######################need to check whether new_solution is already shown or not
                  temp_index = check_whether_it_is_shown(new_solution,all_index_pop_unique)
                  if len(temp_index) == 0:#########no solutions in pop
                      fit_new_solution = fit_train(new_solution, x_train)
                      f_eva = f_eva + 1
                      pop_unique.append(new_solution)
                      pop.append(new_solution)
                      invalid_ind1 = [ind for ind in pop if not ind.fitness.valid]
                      for ind1, fit1 in zip(invalid_ind1, fit_new_solution):
                          ind1.fitness.values = fit1
                      invalid_ind2 = [ind for ind in pop_unique if not ind.fitness.valid]
                      for ind2, fit2 in zip(invalid_ind2, fit_new_solution):
                          ind2.fitness.values = fit2
                  # else:####################new solution has shown before, don't need to do anything
                      # print('temp_index',temp_index)
                      # exit()
                else:# print('this is the second situation')
                  new_solution = combine_all_solutions(ins, feature_possible)
                  temp_index = check_whether_it_is_shown(new_solution, all_index_pop_unique)
                  if len(temp_index) == 0:################new solution should insert the pop_unique
                      fit_new_solution = fit_train(new_solution, x_train)
                      f_eva = f_eva + 1
                      pop_unique.append(new_solution)
                      pop.append(new_solution)
                      invalid_ind1 = [ind for ind in pop if not ind.fitness.valid]
                      for ind1, fit1 in zip(invalid_ind1, fit_new_solution):
                          ind1.fitness.values = fit1
                      invalid_ind2 = [ind for ind in pop_unique if not ind.fitness.valid]
                      for ind2, fit2 in zip(invalid_ind2, fit_new_solution):
                          ind2.fitness.values = fit2
    return pop, f_eva,pop_unique