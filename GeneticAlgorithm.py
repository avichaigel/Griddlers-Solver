import numpy as np
import random
from numpy.random import default_rng
import math
import sys
from time import time

POP_SIZE = 500
NICHE_SIZE = 100
GENS = 200
finish_line_reached = False
rng = default_rng()
FILE = "16.txt"

ranked = []
for i in range(POP_SIZE):
    for j in range(i + 1):
        ranked.append(i)
niche_ranked = []
for n in range(NICHE_SIZE):
    for m in range(n + 1):
        niche_ranked.append(n)

def intialize_pop_gen():
    i = 0
    pop_holder = np.ndarray(POP_SIZE, dtype=object)
    while i < POP_SIZE:
        random_number_of_ones = np.random.randint(0, 626)
        matrix = np.zeros((25, 25))
        indices_to_change = np.random.choice(matrix.size, random_number_of_ones, replace=False)
        matrix.ravel()[indices_to_change] = 1
        pop_holder[i] = [matrix.astype(int), 0]
        i += 1
    return pop_holder


def receive_constraints():
    with open(FILE) as fp:
        data = [list(map(int, line.strip().split(' '))) for line in fp]
        goal = 0
        for list_int in data:
            goal += sum(x > 0 for x in list_int)
        constraints = {}
        for i in range(25):
            constraints['col' + str(i + 1)] = np.asarray(data[i])
        for j in range(25, len(data)):
            constraints['row' + str((j - 25) + 1)] = np.asarray(data[j])
    return constraints, goal


def fitness_by_col(solution, col, num_of_col):
    fill_black_counter = 0
    built_in_constrain = []
    for i in range(25):
        if solution[i][num_of_col] == 0:
            if fill_black_counter > 0:
                built_in_constrain.append(fill_black_counter)
                fill_black_counter = 0
            continue
        else:
            fill_black_counter += 1
    if fill_black_counter > 0:
        built_in_constrain.append(fill_black_counter)

    built_in_constrain = np.asarray(built_in_constrain)
    col = col[col != 0]

    # with order without zeros
    order_fitness_score = 0
    for n, m in zip(col, built_in_constrain):
        if n == m:
            order_fitness_score += 1

    return order_fitness_score


def fitness_by_row(solution, row, num_of_row):
    fill_black_counter = 0
    built_in_constrain = []
    for j in range(25):
        if solution[num_of_row][j] == 0:
            if fill_black_counter > 0:
                built_in_constrain.append(fill_black_counter)
                fill_black_counter = 0
            continue
        else:
            fill_black_counter += 1
    if fill_black_counter > 0:
        built_in_constrain.append(fill_black_counter)

    built_in_constrain = np.asarray(built_in_constrain)
    row = row[row != 0]
    order_fitness_score = 0
    # with order without zeros
    for n, m in zip(row, built_in_constrain):
        if n == m:
            order_fitness_score += 1

    return order_fitness_score


def fitness_function(pop, dic_of_constrain, goal_to_reach):
    global finish_line_reached
    for s in range(len(pop)):
        solution_to_fit = pop[s][0]
        fitness_score = 0
        for l in range(25):
            sub_fit_by_col = fitness_by_col(solution_to_fit, dic_of_constrain["col" + str(l + 1)], l)
            sub_fit_by_row = fitness_by_row(solution_to_fit, dic_of_constrain["row" + str(l + 1)], l)
            fitness_score += (sub_fit_by_col + sub_fit_by_row)
        pop[s][1] = fitness_score
        if fitness_score == goal_to_reach:
            finish_line_reached = True
    return pop


def crossover(parent1, parent2):
    # crossover rate is 80%
    child1, child2 = np.zeros((25, 25), dtype=int), np.zeros((25, 25), dtype=int)
    if random.random() < 0.80:
        for i in range(25):
            for j in range(25):
                if random.random() < 0.5:
                    child1[i][j] = parent1[i][j]
                else:
                    child1[i][j] = parent2[i][j]

                if random.random() < 0.5:
                    child2[i][j] = parent1[i][j]
                else:
                    child2[i][j] = parent2[i][j]
    else:
        child1, child2 = parent1, parent2
    return child1, child2


def mutate(child):
    # mutation rate is 5%
    if np.random.randint(20) == 1:
        indices_to_change = np.random.choice(child.size, 26, replace=False)
        for index in indices_to_change:
            child.ravel()[index] = 1 if child.ravel()[index] == 0 else 0
    return child


def mutation(child1, child2):
    child1, child2 = mutate(child1), mutate(child2)
    return child1, child2


def new_gen(pre_pop):
    new_pop_holder = np.ndarray(pre_pop.size, dtype=object)
    sorted_prev_pop = np.asarray(sorted(pre_pop, key=lambda t: t[1]), dtype=object)
    counter = 0
    # first we are going to preform elitism, by inserting top k solution to new_pop_holder
    # also we are going to delete them from the pre_pop as we leave them untouched
    number_of_elitism = int(pre_pop.size / 50)
    while number_of_elitism != 0:
        new_pop_holder[counter] = sorted_prev_pop[len(sorted_prev_pop) - math.ceil(number_of_elitism / 2)]
        sorted_prev_pop[len(sorted_prev_pop) - math.ceil(number_of_elitism / 2)] = sorted_prev_pop[
            len(sorted_prev_pop) - number_of_elitism]
        number_of_elitism -= 1
        counter += 1
    while counter != pre_pop.size:
        # pick randomly 2 solutions for crossover and mutation
        random_indexes1, random_indexes2 = ranked_build_on_fitness(pre_pop.size)
        parent_sol1 = sorted_prev_pop[random_indexes1]
        parent_sol2 = sorted_prev_pop[random_indexes2]
        crossover_child1, crossover_child2 = crossover(parent_sol1[0], parent_sol2[0])
        mut_child1, mut_child2 = mutation(crossover_child1, crossover_child2)
        new_pop_holder[counter] = [mut_child1, 0]
        counter += 1
        if counter == pre_pop.size:
            break
        new_pop_holder[counter] = [mut_child2, 0]
        counter += 1
    # not forgetting that there is an option, just to copy solutions as is
    return new_pop_holder, sorted_prev_pop[len(sorted_prev_pop) - 1][1]


def ranked_build_on_fitness(pop_size):
    ranked_fit = []
    if pop_size == NICHE_SIZE:
        ranked_fit = niche_ranked
    else:
        ranked_fit = ranked
    place = np.random.choice(len(ranked_fit), 2, replace=False)
    return ranked_fit[place[0]], ranked_fit[place[1]]


def opt(pop, constraints):
    new_fitness_score = 0
    for s in range(len(pop)):
        solution_to_fit = np.copy(pop[s][0])
        fitness_compare = np.copy(pop[s][1])
        indices_to_opt = np.random.choice(solution_to_fit.size, 25, replace=False)
        for index in indices_to_opt:
            solution_to_fit.ravel()[index] = 1 if solution_to_fit.ravel()[index] == 0 else 0
            row_num = index // 25
            col_num = index % 25
            cur_row_fitness = fitness_by_row(pop[s][0], constraints["row" + str(row_num + 1)], row_num)
            cur_col_fitness = fitness_by_col(pop[s][0], constraints["col" + str(col_num + 1)], col_num)
            sub_fit_by_col = fitness_by_col(solution_to_fit, constraints["col" + str(row_num + 1)], row_num)
            sub_fit_by_row = fitness_by_row(solution_to_fit, constraints["row" + str(col_num + 1)], col_num)
            if sub_fit_by_col > cur_col_fitness:
                pop[s][0] = solution_to_fit
                pop[s][1] = fitness_compare + (sub_fit_by_col - cur_col_fitness)
                break
            elif sub_fit_by_row > cur_row_fitness:
                pop[s][0] = solution_to_fit
                pop[s][1] = fitness_compare + (sub_fit_by_row - cur_row_fitness)
                break
    return pop


def Regular_new_gen(population, constraints, finish_line):
    population = fitness_function(population, constraints, finish_line)
    population, best_fit = new_gen(population)
    return population, best_fit


def Darwin_new_gen(population, constraints, finish_line):
    population = fitness_function(population, constraints, finish_line)
    population, best_fit = new_gen(population)
    population = opt(population, constraints)
    return population, best_fit


def Lamark_new_gen(population, constraints, finish_line):
    population = fitness_function(population, constraints, finish_line)
    population = opt(population, constraints)
    population, best_fit = new_gen(population)
    return population, best_fit


start = time()
constraints, finish_line = receive_constraints()
population = intialize_pop_gen()
num_of_gen = 0
flag_for_niching = 0
np.random.shuffle(population)
pop_all_in, niche1, niche2, niche3, niche4, niche5, = [],[],[],[],[],[]
# while num_of_gen < GENS:
while not finish_line_reached:
    if num_of_gen == 0:
        niche1 = population[0:NICHE_SIZE]
        niche2 = population[NICHE_SIZE:2*NICHE_SIZE]
        niche3 = population[2*NICHE_SIZE:3*NICHE_SIZE]
        niche4 = population[3*NICHE_SIZE:4*NICHE_SIZE]
        niche5 = population[4*NICHE_SIZE:5*NICHE_SIZE]
    elif flag_for_niching == 1:
        niche1 = pop_all_in[0:NICHE_SIZE]
        niche2 = pop_all_in[NICHE_SIZE:2*NICHE_SIZE]
        niche3 = pop_all_in[2*NICHE_SIZE:3*NICHE_SIZE]
        niche4 = pop_all_in[3*NICHE_SIZE:4*NICHE_SIZE]
        niche5 = pop_all_in[4*NICHE_SIZE:5*NICHE_SIZE]
        flag_for_niching = 0
    niche1, best1 = Regular_new_gen(niche1, constraints, finish_line)
    niche2, best2 = Regular_new_gen(niche2, constraints, finish_line)
    niche3, best3 = Regular_new_gen(niche3, constraints, finish_line)
    niche4, best4 = Regular_new_gen(niche4, constraints, finish_line)
    niche5, best5 = Regular_new_gen(niche5, constraints, finish_line)
    print("best fit: " + str(100*np.asarray([best1, best2, best3, best4, best5]).max()/finish_line) + '%')
    num_of_gen += 1
    print("gen: " + str(num_of_gen))
    print("")
    # if finish_line_reached:
    #     break
    if num_of_gen % 50 == 0:
        pop_all_in = np.concatenate((niche1, niche2, niche3, niche4, niche5))
        for_100_gen = num_of_gen
        while num_of_gen != for_100_gen + 2:
            pop_all_in, best6 = Regular_new_gen(pop_all_in, constraints, finish_line)
            np.random.shuffle(pop_all_in)
            num_of_gen += 1
        flag_for_niching = 1

end = time()
print("solution found in " + str((start - end)) + " seconds")