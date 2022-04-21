from __future__ import division
import random
import numpy as np
from deap import tools, base
toolbox = base.Toolbox()
import geatpy as ea


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



def hamming_distance(s, s0):
    """Return the Hamming distance between equal-length sequences"""
    s1 = toolbox.clone(s)
    s2 = toolbox.clone(s0)
    s3 = continus2binary(s1)
    s4 = continus2binary(s2)
    if len(s3) != len(s4):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(el1 != el2 for el1, el2 in zip(s3, s4))

def continus2binary(x):
    for i in range(len(x)):
            if x[i] >= 0.6:
                x[i] = 1.0
            else:
                x[i] = 0.0
    return x


def euclidean_distance(x1,x2):
    s1 = toolbox.clone(x1)
    s2 = toolbox.clone(x2)
    s1 = np.array(s1)
    s2 = np.array(s2)
    temp = sum((s1-s2)**2)
    temp1 = np.sqrt(temp)
    return temp1


def remove_empty(pop):
    for i in range(len(pop)):
        x1 = pop[i]
        x = np.zeros((1, len(x1)))
        for ii in range(len(x1)):
            x[0, ii] = x1[ii]
        x = random.choice(x)
        x = 1 * (x >= 0.6)
        if np.count_nonzero(x) == 0:
            kk = random.randint(0,len(pop[0])-1)
            pop[i][kk] = 1
    return pop



def two_CD_one(a,b):
    distances2 = [0.0] * len(a)  #####all is 0
    avg_a = np.mean(a)
    avg_b = np.mean(b)
    for i_i in range(len(a)):
        if a[i_i] > avg_a or b[i_i] > avg_b:
            distances2[i_i] = max(a[i_i],b[i_i])
        else:
            distances2[i_i] = min(a[i_i],b[i_i])
    return distances2





def dis_to_ideal(EXA_0_list,EXA_1_list,EXA_fit,whole_index,sucess_index):
    idea_point = [min(EXA_0_list), min(EXA_1_list)]  #####minimal error rate #####minimal subset size
    dis_to_idea_point = []
    add_index = []
    for i1 in range(len(whole_index)):
        i_temp = whole_index[i1]
        for i2 in range(len(i_temp)):
            if i_temp[i2] not in sucess_index:
                add_index.append(i_temp[i2])
                dis_to_idea_point.append(euclidean_distance(idea_point, EXA_fit[i_temp[i2]]))
    return dis_to_idea_point,add_index


def dis_to_each_dimension_lowest(dis,whole_index,sucess_index):
    need_compared_dis = []
    add_index = []
    for i1 in range(len(whole_index)):
      i_temp = whole_index[i1]
      for i2 in range(len(i_temp)):
          if i_temp[i2] not in sucess_index:
            need_compared_dis.append(dis[i1][i2])
            add_index.append(i_temp[i2])
    return need_compared_dis,add_index


def crowding_estimation_in_search_space(pop,s):
    NDIM = len(pop[0])
    EXA = s
    used_set = pop
    ham_dis = np.zeros((len(EXA), len(pop)))
    distances_new = [0.0] * len(EXA)
    distances_new1 = [0.0] * len(EXA)
    # norm = 0
    for i in range(len(EXA)):
        for j in range(len(used_set)):
            ham_dis[i, j] = hamming_distance(EXA[i], used_set[j]) / NDIM  ####the position of 0 is changing
        sorts1 = sorted(ham_dis[i, :], reverse=False)  ###sorting from minimum to maximum
        sorts = sorts1[1:]  ####the first one is 0, because it's itself.
        nei_num = min(8, len(sorts))  ####the number of particle
        # for j10 in range(nei_num):
        #     norm += float(nei_num * sorts[j10])
        for j1 in range(nei_num):
            distances_new[i] += (nei_num - j1) * sorts[j1]  ###(nei_num − j + 1)d ij
            # distances_new1[i] += (nei_num - j1) * sorts[j1] / norm
    distances_new = np.array(distances_new) / np.array(max(distances_new))
    # distances_new1 = np.array(distances_new1) / np.array(max(distances_new1))
    return distances_new



def assignment_in_objective(pop,ee):########the multimmodal solutions have the same crowding score
    distances = [10] * len(pop)
    whole_index = []########################each dimension has num solutions
    multimal_index = []#######the set of index
    dis = []
    cal_index = []
    noncal_index = []
    EXA_fit = np.array([ind.fitness.values for ind in pop])
    EXA_0_list = list(EXA_fit[:, 0])  #####error rate
    EXA_1_list = list(EXA_fit[:, 1])  #####subset size
    single1_index = set(EXA_1_list)
    single1_index = sorted(list(single1_index))
    # print('the total groups', len(single1_index),len(pop))
    if len(single1_index) == len(pop):  #### this means the solutions have different sizes that means they are all unique.
        distances = crowding_estimation_in_objective_space(pop)
        return distances
    else:  ###############according to the size index to find the error within the range ee,
        for i in range(len(single1_index)):
            index_of_size = findindex(EXA_fit[:, 1],single1_index[i]) ###the index of some solutions with the same size(size changes)
            whole_index.append(index_of_size)
            if len(index_of_size) > 1:  ###multiple solutions have the same size
                temp0 = [EXA_0_list[m] for m in index_of_size]
                temp_minus_min = [abs(jj-min(temp0)) for jj in temp0]
                dis.append(temp_minus_min)
                index_error = np.argwhere(abs(temp0-min(temp0)) <= ee)##########the distance to the minimal error
                if len(index_error) > 1:#######multiple different solutions have similar classification performance
                    list1 = []
                    for ii in index_error:
                        iii = random.choice(ii)
                        list1.append(iii)
                    list2 = [index_of_size[t] for t in list1]  ####size same, and error in a range same
                    multimal_index.append(list2)
            else:
                ccc = random.choice(index_of_size)
                cal_index.append(ccc)
        ####get an index set where the solutions need to calculate the crowding distance
        for i in multimal_index:
            cal_index.append(i[0])
        need_claculate_individuals = [pop[mm] for mm in cal_index]
        temp_distances = crowding_estimation_in_objective_space(need_claculate_individuals)
        for i_ii in range(len(cal_index)):
            distances[cal_index[i_ii]] = temp_distances[i_ii]
        for i_iii in multimal_index:
            for ij in range(1,len(i_iii)):
                distances[i_iii[ij]] = distances[i_iii[0]]
    return distances



def crowding_estimation_in_objective_space(individuals):
    if len(individuals) == 0:
        return
    distances = [0.0] * len(individuals)
    crowd = [(ind.fitness.values, i) for i, ind in enumerate(individuals)]
    nobj = 2
    for i in range(nobj):
        crowd.sort(key=lambda element: element[0][i])
        distances[crowd[0][1]] = 0
        distances[crowd[-1][1]] = 1
        if crowd[-1][0][i] == crowd[0][0][i]:
            continue
        norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])
        for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]] += (next[0][i] - prev[0][i]) / norm
    return distances



def output(pop,ee):########################output the solutions without using non-diminated concept
    s = []#########################the output
    whole_index = []########################each dimension has num solutions
    dis = []
    if len(pop) == 0:
        return pop
    EXA_fit = np.array([ind.fitness.values for ind in pop])
    EXA_0_list = list(EXA_fit[:, 0])  #####error rate
    EXA_1_list = list(EXA_fit[:, 1])  #####subset size
    single1_index = set(EXA_1_list)
    single1_index = sorted(list(single1_index))
    for i in range(len(single1_index)):
        index_of_size = findindex(EXA_fit[:, 1],single1_index[i]) ###the index of some solutions with the same size(size changes)
        if len(index_of_size) > 1:  ###multiple solutions have the same size
                whole_index.append(index_of_size)
                temp0 = [EXA_0_list[m] for m in index_of_size]
                temp_minus_min = [abs(jj-min(temp0)) for jj in temp0]
                dis.append(temp_minus_min)
                index_error = np.argwhere(abs(temp0-min(temp0)) <= ee)##########the distance to the minimal error
                if len(index_error) > 1:#######multiple different solutions have similar classification performance
                    list1 = []
                    for ii in index_error:
                        iii = random.choice(ii)
                        list1.append(iii)
                    list2 = [index_of_size[t] for t in list1]  ####size same, and error in a range same
                    ins = [pop[tt] for tt in list2]
                    s.extend(ins)
                else:#(len(index_error) == 1) didn't have multimidal solutions
                    c = random.choice(index_error)
                    cc = random.choice(c)
                    s.append(pop[index_of_size[cc]])
        else:###########################only one solution select num features
                ccc = random.choice(index_of_size)
                s.append(pop[ccc])
    fit = np.array([ind.fitness.values for ind in s])
    return s, fit


def output2(pop,ee):########################output the solutions using non-diminated concept
    whole_index = []  ########################each dimension has num solutions
    multimal_index = []  #######the set of index
    dis = []
    EXA_fit = np.array([ind.fitness.values for ind in pop])
    EXA_0_list = list(EXA_fit[:, 0])  #####error rate
    EXA_1_list = list(EXA_fit[:, 1])  #####subset size
    single1_index = set(EXA_1_list)
    single1_index = sorted(list(single1_index))
    if len(single1_index) == len(pop):  #### this means the solutions have different sizes that means they are all unique.
        distances = crowding_estimation_in_objective_space(pop)
        return distances
    else:  ###############according to the size index to find the error within the range ee,
        for i in range(len(single1_index)):
            index_of_size = findindex(EXA_fit[:, 1],single1_index[i])  ###the index of some solutions with the same size(size changes)
            whole_index.append(index_of_size)
            if len(index_of_size) > 1:  ###multiple solutions have the same size
                temp0 = [EXA_0_list[m] for m in index_of_size]
                temp_minus_min = [abs(jj - min(temp0)) for jj in temp0]
                dis.append(temp_minus_min)
                index_error = np.argwhere(abs(temp0 - min(temp0)) <= ee)  ##########the distance to the minimal error
                if len(index_error) > 1:  #######multiple different solutions have similar classification performance
                    list1 = []
                    for ii in index_error:
                        iii = random.choice(ii)
                        list1.append(iii)
                    list2 = [index_of_size[t] for t in list1]  ####size same, and error in a range same
                    multimal_index.append(list2)
    out_index = []
    index_non = first_nondominated(pop)
    for i in index_non:
        for j in range(len(multimal_index)):
            if i in multimal_index[j]:
                for ij in multimal_index[j]:
                   out_index.append(ij)
            else:
                out_index.append(i)
    out_index = list(set(out_index))
    s = [pop[m] for m in out_index]
    fit = np.array([ind.fitness.values for ind in s])
    return s, fit



def first_nondominated(pop):
    PF = np.array([ind.fitness.values for ind in pop])
    [levels1, criLevel1] = ea.indicator.ndsortDED(PF, 1)
    x1 = 1 * (levels1 == 1.0)
    x1 = "".join(map(str, x1))
    index_non = np.array(list(find_all(x1, '1')))
    return index_non
