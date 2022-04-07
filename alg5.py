from __future__ import division
import bisect
import math
import random
from itertools import chain
from operator import attrgetter, itemgetter
from collections import defaultdict
import numpy as np
from deap import base
import geatpy as ea
import matplotlib.pyplot as plt
#############################################################ICD from yue + my ideas
################################increase from 0.5~1 in generation 1~0.5max_gen

def more_confidence(EXA, index_of_objectives):
    a = 0.6
    cr = np.zeros((len(index_of_objectives),1))
    for i in range(len(index_of_objectives)):###the number of indexes
        temp = 0
        object = EXA[index_of_objectives[i]]
        for ii in range(len(object)):###the number of features
           b = object[ii]
           if b > a:  con = (b - a) / (1 - a)
           else:      con = (a - b) / (a)
           temp = con + temp
        cr[i,0] = temp
    sorting = np.argsort(-cr[:,0])####sorting from maximum to minimum
    index_one = index_of_objectives[sorting[0]]
    return index_one


def delete_duplicate(EXA):####list
    EXA1 = []
    EXA_array = np.array(EXA)
    all_index = []
    for i0 in range(EXA_array.shape[0]):
       x = 1 * (EXA_array[i0,:] >= 0.6)
       x = "".join(map(str, x))  # transfer the array form to string in order to find the position of 1
       all_index.append(x)##store all individuals who have changed to 0 or 1
    single_index = set(all_index)####find the unique combination
    single_index = list(single_index)####translate it's form in order to following operating.
    for i1 in range(len(single_index)):
       index_of_objectives = findindex(all_index, single_index[i1])##find the index of each unique combination
       if len(index_of_objectives) == 1:
          for i2 in range(len(index_of_objectives)):
             EXA1.append(EXA[index_of_objectives[i2]])
       else:####some combination have more than one solutions.here may have duplicated solutions
           index_one = more_confidence(EXA, index_of_objectives)
           EXA1.append(EXA[index_one])
    return EXA1


def two_CD_one(a,b,rank):
    distances2 = [0.0] * len(a)  #####all is 0
    avg_a = np.mean(a)
    avg_b = np.mean(b)
    for i_i in range(len(a)):
        if a[i_i] > avg_a or b[i_i] > avg_b:
            distances2[i_i] = max(a[i_i],b[i_i])#/(rank +1))
        else:
            distances2[i_i] = min(a[i_i],b[i_i])
    return distances2




def weight_one(a,b,rank,gen):#####a is in decision space, while b is in objective space
    distances2 = [0.0] * len(a)  #####all is 0
    for i_i in range(len(a)):
        # distances2[i_i] = (0.4+(gen+1)/200)*a[i_i]+(0.6-(gen+1)/200)*b[i_i]###/(rank +1)
        distances2[i_i] = (0.7 - 0.7*math.exp(-0.05*gen)) * a[i_i] + (0.3 +0.7*math.exp(-0.05*gen)) * b[i_i]  ###/(rank +1)
        # distances2[i_i] = 1/(rank +1) + 0.1*a[i_i] +  0.35 *b[i_i]
    return distances2


def DC(a, b, rank): ##distances_ds, distances_os
    distances3 = [0.0] * len(a)  #####all is 0
    for i_i in range(len(a)):
        # distances3[i_i] = 1/(rank +1) + 0.35* b[i_i] + 0.1*a[i_i]
        distances3[i_i] = 0.35 * b[i_i] + 0.1 * a[i_i]
    return distances3






def continus2binary(x):
    for i in range(len(x)):
            if x[i] >= 0.6:
                x[i] = 1.0
            else:
                x[i] = 0.0
    return x

toolbox = base.Toolbox()

def hamming_distance(s, s0):
    """Return the Hamming distance between equal-length sequences"""
    s1 = toolbox.clone(s)
    s2 = toolbox.clone(s0)
    s3 = continus2binary(s1)
    s4 = continus2binary(s2)
    if len(s3) != len(s4):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(el1 != el2 for el1, el2 in zip(s3, s4))



def euc_distance(a,b):
    sum = 0
    for i in range(len(a)):
        sum += (a[i]-b[i])**2
    dis = math.sqrt(sum)
    return dis


def crowding_in_solution_space1(EXA,used_set):
    distances_new = [0.0] * len(EXA)  #####all is 0
    k = len(EXA[0])
    ham_dis = np.zeros((len(EXA), len(used_set)))
    if len(EXA) == 1:
        distances_new[0] = 1
    else:
        for i in range(len(EXA)):
            for j in range(len(used_set)):
                    ham_dis[i, j] = hamming_distance(EXA[i], used_set[j]) / k  ####the position of 0 is changing
            sorts1 = sorted(ham_dis[i, :], reverse=False)  ###sorting from minimum to maximum
            sorts = sorts1[1:]  ####the first one is 0, because it's itself.
            nei_num = min(8, len(sorts))  ####the number of particle
            for j1 in range(nei_num):
                distances_new[i] += (nei_num - j1) * sorts[j1]  ###(nei_num − j + 1)d ij
    distances_new = np.array(distances_new)/np.array(max(distances_new))
    return distances_new



def crowding_in_solution_space(EXA,used_set):###################no multiply weight
    distances_new = [0.0] * len(EXA)  #####all is 0
    k = len(EXA[0])
    ham_dis = np.zeros((len(EXA), len(used_set)))
    if len(EXA) == 1:
        distances_new[0] = 1
    else:
        for i in range(len(EXA)):
            for j in range(len(used_set)):
                    ham_dis[i, j] = hamming_distance(EXA[i], used_set[j]) / k  ####the position of 0 is changing
            sorts1 = sorted(ham_dis[i, :], reverse=False)  ###sorting from minimum to maximum
            sorts = sorts1[1:]  ####the first one is 0, because it's itself.
            nei_num = min(8, len(sorts))  ####the number of particle
            for j1 in range(nei_num):
                # distances_new[i] += (nei_num - j1) * sorts[j1]  ###(nei_num − j + 1)d ij
                distances_new[i] += sorts[j1]/nei_num
        distances_new = np.array(distances_new)/np.array(max(distances_new))
    return distances_new




def crowding_in_objective_space(EXA,used_set):
    EXA_fit = np.array([ind.fitness.values for ind in EXA])
    distances_new = [0.0] * len(EXA)#####all is 0
    k = 2
    crowd = [(ind.fitness.values, i) for i, ind in enumerate(EXA)]  ######fitness value
    EXA_matrix0 = np.zeros((len(EXA), k))
    EXA_matrix1 = np.zeros((len(used_set), k))
    EXA_fit1 = np.array([ind.fitness.values for ind in used_set])
    for j in range(k):  #### k objective variables
            crowd.sort(key=lambda element: element[0][j])  ####sorting based on the j objective)
            for i in range(len(EXA)):
                EXA_matrix0[i, :] = EXA_fit[i]
            for ii in range(len(used_set)):
                EXA_matrix1[ii, :] = EXA_fit1[ii]
            sortt = sorted(EXA_matrix1[:, j])
            if crowd[-1][0][j] == crowd[0][0][j]:
                continue
            norm = k * float(sortt[-1] - sortt[0])
            if norm == 0:
                norm = 1
            for i in range(len(crowd)):
                index = findindex(sortt, EXA_matrix0[i, j])
                index = random.choice(index)
                if index == 0:
                    distances_new[i] += 1
                elif index == len(sortt) - 1:
                    distances_new[i] += 0
                else:
                    distances_new[i] += (sortt[index + 1] - sortt[index - 1]) / norm
    return distances_new


def find_non_sorting(EXA_fit):
    [levels1, criLevel1] = ea.indicator.ndsortDED(EXA_fit)
    whole_rank = list(set(levels1))
    each_rank_index = []
    for i in whole_rank:
        x1 = 1 * (levels1 == i)
        x1 = "".join(map(str, x1))
        index_non = list(find_all(x1, '1'))
        each_rank_index.append(index_non)
    return each_rank_index


######################################
# main
######################################
def selNS(pop, ee, MU, gen):####len(individuals) = len(pop) = k
    each_rank_index,multimal_index = find_multimodal_solutions(pop,ee)
    if multimal_index == []:
        s = no_mo_solutions(pop, each_rank_index, MU,gen)
    else:
        s = have_mo_solutions(pop, multimal_index, each_rank_index, MU,gen)
    return s,each_rank_index





def no_mo_solutions(pop,each_rank_index,MU,gen):
    used_set_index = []
    in2 = 0
    for in1 in range(len(each_rank_index)):
        rank_sort = each_rank_index[in1]
        in2 = in2 + len(rank_sort)
        if in2 < MU:  ###smaller the pop size
            used_set_index.extend(rank_sort)
        elif in2 == MU:  # reach the pop size,print('dont need calculate the crowding distance')
            used_set_index.extend(rank_sort)
            s = [pop[m] for m in used_set_index]
            return s
        else:  ##larger the pop size, need remove solutions
            # print(used_set_index)
            # exit()
            rank = in1 + 1
            need_remove_num = in2 - MU
            #############################################################in search/decision space, based on nolvety score
            used_set_index1 = [k for k in used_set_index]
            used_set_index1.extend(rank_sort)  ####
            pop_need = [pop[k] for k in rank_sort]
            used_set = [pop[k] for k in used_set_index1]
            dis_ds = crowding_in_solution_space(pop_need, pop)
            ################################################################################in objective space
            dis_os = crowding_in_objective_space(pop_need, used_set)
            # scd = DC(dis_ds, dis_os, rank)
            scd = two_CD_one(dis_ds, dis_os, rank)
            # scd = weight_one(dis_ds, dis_os, rank, gen)
            index3 = np.argsort(scd)
            delete_index = [rank_sort[k] for k in index3[:need_remove_num]]
            [rank_sort.remove(m) for m in delete_index]
            used_set_index.extend(rank_sort)
            s = [pop[m] for m in used_set_index]
            return s


def have_mo_solutions(pop,multimal_index,each_rank_index,MU,gen):
    used_set_index = []
    multimal_index_list = []
    for i in multimal_index:
        for j in i:
            multimal_index_list.append(j)
    in2 = 0
    for in7 in range(len(each_rank_index)):
        rank_sort = each_rank_index[in7]
        in2 = in2 + len(rank_sort)
        if in2 < MU:  ###smaller the pop size
            used_set_index.extend(rank_sort)
        elif in2 == MU:  # reach the pop size,print('dont need calculate the crowding distance')
            used_set_index.extend(rank_sort)
            s = [pop[m] for m in used_set_index]
            return s
        else:  ##larger the pop size, need remove solutions
            # print(used_set_index)
            # exit()
            need_remove_num = in2 - MU
            set0 = []  ###store different solutions
            set1 = []  ##store multimodal solutions
            rank = in7 + 1
            for i in range(len(rank_sort)):
                if rank_sort[i] in multimal_index_list:
                    set1.append(rank_sort[i])
                else:
                    set0.append(rank_sort[i])
            index4 = set0 + set1
            if len(set1) == 0:  ####in the third front, no multimodal solutions
                used_set_index1 = [k for k in used_set_index]
                used_set_index1.extend(rank_sort)  ####
                pop_need = [pop[k] for k in rank_sort]
                used_set = [pop[k] for k in used_set_index1]
                dis_ds = crowding_in_solution_space(pop_need, pop)
                ################################################################################in objective space
                dis_os = crowding_in_objective_space(pop_need, used_set)
                # scd = DC(dis_ds, dis_os, rank)
                scd = two_CD_one(dis_ds, dis_os, rank)
                # scd = weight_one(dis_ds, dis_os, rank, gen)
                index3 = np.argsort(scd)
                delete_index = [rank_sort[k] for k in index3[:need_remove_num]]
                [rank_sort.remove(m) for m in delete_index]
                used_set_index.extend(rank_sort)
                s = [pop[m] for m in used_set_index]
                return s
            else:  ##two sets, one stores the needed calculation, and another one stores the remaing one
                #############################################################in search/decision space, based on nolvety score
                used_set_index1 = [k for k in used_set_index]
                used_set_index1.extend(rank_sort)  ####
                pop_need1 = [pop[k] for k in rank_sort]
                used_set1 = [pop[k] for k in used_set_index1]
                dis_ds = crowding_in_solution_space(pop_need1, pop)
                ################################################################################in objective space
                set00 = [k for k in set0]  ####need put it all
                set00.append(set1[0])
                used_set_index2 = [k for k in used_set_index]
                used_set_index2.extend(set00)  ####
                pop_need = [pop[k] for k in set00]  ################only include one multimodal solution
                used_set = [pop[k] for k in used_set_index2]
                dis_part = crowding_in_objective_space(pop_need, used_set)
                [dis_part.append(dis_part[-1]) for _ in range(1, len(set1))]
                # scd = DC(dis_ds, dis_part, rank)
                scd = two_CD_one(dis_ds, dis_part, rank)
                # scd = weight_one(dis_ds, dis_part, rank, gen)
                index3 = np.argsort(scd)
                delete_index = [index4[k] for k in index3[:need_remove_num]]
                [rank_sort.remove(m) for m in delete_index]
                used_set_index.extend(rank_sort)
                s = [pop[m] for m in used_set_index]
                return s



def find_multimodal_solutions(pop,ee):#len(single1_index)<len(pop)
    EXA_fit = np.array([ind.fitness.values for ind in pop])
    multimal_index = []
    EXA_0_list = list(EXA_fit[:, 0])  #####error rate
    EXA_1_list = list(EXA_fit[:, 1])  #####subset size
    single1_index = set(EXA_1_list)
    single1_index = sorted(list(single1_index))
    ###############################################obtain the rank sorting
    each_rank_index = find_non_sorting(EXA_fit)
    # print('before',each_rank_index)
    ###############################################obtain the rank sorting
    if len(single1_index) == len(pop):  #### this means the solutions have different sizes that means they are all unique.
        each_rank_index = [x for x in each_rank_index if x != []]
    else:  ###according to the size index to find the error within the range ee,
       for i in range(len(single1_index)):
         index_of_size = findindex(EXA_fit[:, 1],single1_index[i])###the index of some solutions with the same size(size changes)
         if len(index_of_size) > 1:###multiple solutions have the same size
            temp0 = [EXA_0_list[m] for m in index_of_size]
            index_error = np.argwhere(abs(temp0 - min(temp0)) <= ee)  ##########the distance to the minimal error
            if len(index_error) > 1:  #######multiple different solutions have similar classification performance
               each_rank_index,multimal_index = find_error_same_with_non_sorting(index_error,index_of_size,multimal_index,each_rank_index,pop,single1_index)
       each_rank_index = [x for x in each_rank_index if x != []]
    # print('multimal_index',multimal_index)
    # for i in multimal_index:
    #     pop1 = [pop[k] for k in i]
    #     print(pop1)
    #     print(np.array([ind.fitness.values for ind in pop1]))
    # print('after',each_rank_index)
    return each_rank_index,multimal_index


def determination_limit_number(each_rank_index,pop):
    first_rank_pop = [pop[m] for m in each_rank_index[0]]
    EXA_fit = np.array([ind.fitness.values for ind in pop])
    EXA_1_list = list(EXA_fit[:, 1])  #####subset size
    single1_index = set(EXA_1_list)
    num = len(pop) // len(single1_index)
    return num




def find_error_same_with_non_sorting(index_error,index_of_size,multimal_index,each_rank_index,pop,single1_index):
    list1 = []
    for ii in index_error:
        iii = random.choice(ii)
        list1.append(iii)
    list2 = [index_of_size[t] for t in list1]  ####size same, and error in a range same
    # limit = determination_limit_number(each_rank_index,pop)
    limit = len(pop)//len(single1_index)
    if len(list2) > limit:
        ###########################################one way is based on the crowding distance in solution space
        p1 = [pop[m] for m in list2]
        distances = crowding_in_solution_space(p1, pop)
        sorting_index = np.argsort(-distances)
        save_index = sorting_index[:limit]
        list2 = [list2[t] for t in save_index]
        ###########################################another way is randomly choosing, bad
        # list2 = list2[:limit]
        #####################################another way is based on redundancy rat
    multimal_index.append(list2)
    in1 = [0] * len(each_rank_index)  ####the idea is to find the position of the multimoal solutions in the front rank set
    for i1 in list2:  ##multimodal index
        for i2 in range(len(each_rank_index)):
            each_rank = each_rank_index[i2]  ###
            if i1 in each_rank:
                in1[i2] = in1[i2] + 1
    in2 = np.argwhere(np.array(in1) > 0)  ####
    if len(in2) > 1:###############the indexes are locating at different fronts
        in4 = random.choice(in2[0])
        each_rank_index[in4].extend(list2)
        each_rank_index[in4] = list(set(each_rank_index[in4]))
        ############from the second position to delete index
        for in3 in range(1, len(in2)):
            in5 = random.choice(in2[in3])
            for in6 in list2:
                if in6 in each_rank_index[in5]:
                    each_rank_index[in5].remove(in6)
    ########if the indexes are locating at the same front, in the first front, there may a lot of duplicated points.
    # else:  ##also need to limit the number
    #     print(in2)
    #     print(list2)
    #     print(limit)
    #     exit()
    return each_rank_index,multimal_index


def findindex(org, x):
    result = []
    for k,v in enumerate(org): #k和v分别表示org中的下标和该下标对应的元素
        if v == x:
            result.append(k)
    return result


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)