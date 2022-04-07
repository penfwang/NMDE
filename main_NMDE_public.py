from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
import array
import random
import numpy as np
from deap import base
import math
from deap.benchmarks.tools import hypervolume
from deap import creator
from deap import tools
import alg5
import operator
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import geatpy as ea
from diverse import improved_evaluation
from local_search import subset_repairing_scheme
from new_selection import remove_empty
# from initialization import DAEA_initialization


def uniform(low, up, size=None):####generate a matrix of the range of variables
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

def findindex(org, x):
    result = []
    for k,v in enumerate(org): 
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
     # clf = SVC(kernel="linear", C=0.025)
     # clf = DecisionTreeClassifier(max_depth=5)
     # clf = MLPClassifier(alpha=1, max_iter=1000)

     scores = cross_val_score(clf, tr[:,1:],tr[:,0], cv = 10)
     f1 = np.mean(1 - scores)
     f2 = (len(value_position)-1)/(train_data.shape[1] - 1)
     # f2 = len(value_position) - 1
    return f1, f2

def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]   
    diff = np.tile(newInput, (numSamples, 1)) - dataSet  
    squaredDiff = diff ** 2  
    squaredDist = squaredDiff.sum(axis = 1)   
    distance = squaredDist ** 0.5 
    sortedDistIndices = distance.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sorted_ClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_ClassCount[0][0]


def evaluate_test_data(x, train_data, test_data):
    x = 1 * (x >= 0.6)
    x = "".join(map(str, x))  # transfer the array form to string in order to find the position of 1
    value_position = np.array(list(find_all(x, '1'))) + 1  # cause the label in the first column in training data
    value_position = np.insert(value_position, 0, 0)  # insert the column of label
    te = test_data[:, value_position]#####testing data including label in the first colume
    tr = train_data[:, value_position]#####training data including label in the first colume too
    wrong = 0
    for i12 in range(len(te)):
        testX = te[i12,1:]
        dataSet = tr[:,1:]
        labels = tr[:,0]
        outputLabel = kNNClassify(testX, dataSet, labels, 5)
        # print(outputLabel,te[i12,0])
        if outputLabel != te[i12,0]:
            wrong = wrong + 1
    f1 = wrong/len(te)
    f2 = (len(value_position) - 1) / (test_data.shape[1] - 1)
    return f1, f2


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


def mutDE(y, a, b, c, f):###mutation:DE/rand/1; if a is the best one, it will change as DE/best/1
    for i in range(len(y)):
        y[i] = a[i] + f*(b[i]-c[i])
    return y


def proposed_NGI(niche_offspring,ss,offspring,member,ii,f_mu):##only use the non-dominated solutions in the niche
    index_non1 = first_nondominated(niche_offspring)  ##to check whether the current individual is in the first front
    temp_index = [ss[jj] for jj in index_non1]
    y_new = toolbox.clone(member)
    temp_individual = [offspring[i] for i in temp_index]
    temp_individual_fitness = np.array([ind.fitness.values for ind in temp_individual])
    index1 = np.argsort(temp_individual_fitness[:, 0])  ###error
    index1 = random.choice(index1)
    nbest = temp_individual[index1]
    if ii in temp_index:  ####means individual ii is in the first front
        r1, r2 = random.sample(niche_offspring, 2)  ####two uniuqe individuals
        for i2 in range(len(y_new)):
            y_new[i2] = member[i2] + f_mu * (r1[i2] - r2[i2])
    else:  ########if other members (not ii) in the first front, randomly choose one as nbest,
        offspring1 = toolbox.clone(offspring)
        offspring1.remove(member)
        offspring1.remove(nbest)
        in1, in2 = random.sample(offspring1, 2)
        for i2 in range(len(y_new)):
            y_new[i2]=member[i2] +f_mu*(nbest[i2]-member[i2])+f_mu*(in1[i2]-in2[i2])
    return y_new,nbest



def cxBinomial(x, y, cr):#####binary crossover
    y_new = toolbox.clone(x)
    size = len(x)
    index = random.randrange(size)
    for i in range(size):
        if i == index or random.uniform(0, 1) <= cr:
            y_new[i] = y[i]
    return y_new


def first_nondominated(pop):
    PF = np.array([ind.fitness.values for ind in pop])
    [levels1, criLevel1] = ea.indicator.ndsortDED(PF, 1)
    x1 = 1 * (levels1 == 1.0)
    x1 = "".join(map(str, x1))
    index_non = np.array(list(find_all(x1, '1')))
    return index_non


def continus2binary(x):
    for i in range(len(x)):
            if x[i] >= 0.6:
                x[i] = 1.0
            else:
                x[i] = 0.0
    return x


def hamming_distance(s, s0):
    """Return the Hamming distance between equal-length sequences"""
    s1 = toolbox.clone(s)
    s2 = toolbox.clone(s0)
    s3 = continus2binary(s1)
    s4 = continus2binary(s2)
    if len(s3) != len(s4):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(el1 != el2 for el1, el2 in zip(s3, s4))


creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))####minimise two objectives
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
toolbox = base.Toolbox()


def main(dataset_name):
    seed = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    random.seed(seed[19])
    folder1 = '/vol/grid-solar/sgeusers/wangpeng/multi-result/split_73' + '/' + 'train' + str(
        dataset_name) + ".npy"
    x_train = np.load(folder1)
    ee = 1 / x_train.shape[0]
    NDIM = x_train.shape[1] - 1
    BOUND_LOW, BOUND_UP = 0.0, 1.0
    NGEN = 100###the number of generation
    if NDIM < 300:
        MU = NDIM  ####the number of particle
    else:
        MU = 300  #####bound to 300
    Max_FES = MU * 100
    # toolbox.register("attr_float", bytes, BOUND_LOW, BOUND_UP, NDIM)
    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)  #####dertemine the way of randomly generation and gunrantuu the range
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)  ###fitness
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  ##particles
    toolbox.register("evaluate", fit_train, train_data= x_train)
    toolbox.register("select1", alg5.selNS)
    offspring = toolbox.population(n=MU)
    # offspring = DAEA_initialization(offspring)
    offspring = remove_empty(offspring)
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)#####toolbox.evaluate = fit_train
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    fit_num = len(offspring)
    unique_number = []
    pop_u = delete_duplicate(offspring)
    unique_number.append(len(pop_u))
    dis = np.zeros((MU,MU))
    for i in range(MU):
        for j in range(MU):
            dis[i,j] = hamming_distance(offspring[i],offspring[j])/NDIM
    for gen in range(1, NGEN):
        print('gen',gen)
        pop_new = toolbox.clone(offspring)
        for ii in range(MU):#####upate the whole population
            ss_i = np.argsort(dis[ii, :])  ###the index in the whole population
            niche_offspring = [offspring[t] for t in ss_i[:9]]
            ####################################the niching, local binary pattern
            y_new, nbest = proposed_NGI(niche_offspring,ss_i,offspring,offspring[ii],ii,0.5)
            if fit_num <0.7* Max_FES:
                for i_z in range(len(y_new)):
                    if y_new[i_z] > 1:
                       y_new[i_z] = 1
                    if y_new[i_z] < 0:
                       y_new[i_z] = 0
            else:
                for i_z in range(len(y_new)):
                    if y_new[i_z] > 1:
                        y_new[i_z] = nbest[i_z]
                    if y_new[i_z] < 0:
                        y_new[i_z] = nbest[i_z]
            pop_new[ii] = cxBinomial(offspring[ii],y_new,0.5)###crossover
            pop_new = remove_empty(pop_new)
            del pop_new[ii].fitness.values
        ##################################################
        ##IF one solution has shown before, then its fitness values will be taken from pop_u
        fit_improved, fit_num = improved_evaluation(pop_new,pop_u,fit_num,x_train)##<class 'numpy.ndarray'>
        invalid_ind = [ind for ind in pop_new if not ind.fitness.valid]
        for ind, fit in zip(invalid_ind, fit_improved):
            ind.fitness.values = fit
        ##############################################################real calculation
        # invalid_ind = [ind for ind in pop_new if not ind.fitness.valid]
        # fitne = toolbox.map(toolbox.evaluate, invalid_ind)
        # for ind, fit1 in zip(invalid_ind, fitne):
        #     ind.fitness.values = fit1
        # fit_num = fit_num + len(offspring)
        ##############################################################real calculation
        pop_u.extend(delete_duplicate(offspring))
        pop_u  = delete_duplicate(pop_u )
        unique_number.append(len(pop_u))
        print(len(pop_u))
        pop_mi = pop_new + offspring
        pop1 = delete_duplicate(pop_mi)
        pop1, fit_num,pop_unique = subset_repairing_scheme(pop1,pop_u,fit_num,x_train,ee)
        offspring, each_rank_index = toolbox.select1(pop1, ee, MU, gen)####alg5
        first_rank_population = [pop1[m] for m in each_rank_index[0]]
        print('len(offspring)', len(offspring))
        for i in range(MU):
            for j in range(MU):
                dis[i, j] = hamming_distance(offspring[i], offspring[j]) / NDIM
        ##########################################################new
        if fit_num > Max_FES:
            break
    print("Final population hypervolume is %f" % hypervolume(offspring, [1, 1]))
    return offspring,first_rank_population


if __name__ == "__main__":
    tt = ['dataSet_zoo']
    dataset = tt[0]
    pop,first_rank_population = main(dataset)
    index_non33 = first_nondominated(pop)
    # front_non = [pop[m] for m in index_non33]
    folder1 = 'train' + str(dataset) + ".npy"
    folder2 = 'test' +  str(dataset) + ".npy"
    x_train = np.load(folder1)
    x_test = np.load(folder2)
    # front_non, front = output2(pop,ee)
    front_non = first_rank_population
    front = np.array([ind.fitness.values for ind in front_non])
    EXA_array = np.array(front_non)
    EXA_01 = 1 * (EXA_array >= 0.6)
    front_testing2 = np.ones((EXA_array.shape[0], 2))
    for i in range(EXA_array.shape[0]):
        front_testing2[i, :] = evaluate_test_data(EXA_array[i, :], x_train, x_test)
    name1 = 'NSGAII_train_'+ str(dataset) + ".npy"
    f = np.load(name1)
    fig1 = plt.scatter(front[:,1], front[:,0], color='r', lw=1.5,marker='o',ls= '-.')
    fig2 =plt.scatter(f[:, 1], f[:, 0], color='k', lw=1.5,marker='+',ls= '-.')
    plt.grid(True)
    plt.legend([fig1,fig2],['NMDE','NSGAII'])
    plt.axis("tight")
    plt.show()
    name2 = 'NSGAII_test_'+ str(dataset) + ".npy"
    f1 = np.load(name2)
    fig1 = plt.scatter(front_testing2[:,1], front_testing2[:,0], color='r', lw=1.5,marker='o',ls= '-.')
    fig2 =plt.scatter(f1[:, 1], f1[:, 0], color='k', lw=1.5,marker='+',ls= '-.')
    plt.grid(True)
    plt.legend([fig1,fig2],['NMDE','NSGAII'])
    plt.axis("tight")
    plt.show()
