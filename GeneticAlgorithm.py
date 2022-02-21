import numpy as np
import random
from numpy.random import default_rng
import math
from time import time

import Plot

DELIMITER = "#\n"
POP_SIZE = 1000
NICHE_NUM = 5
NICHE_SIZE = POP_SIZE//NICHE_NUM
TOTAL_GENS = 100
optimal_solution_found = False
rng = default_rng()
row_num = 0
col_num = 0

# create rank lists, for the selection process
ranked = []
for i in range(POP_SIZE):
    for j in range(i + 1):
        ranked.append(i)
niche_ranked = []
for n in range(NICHE_SIZE):
    for m in range(n + 1):
        niche_ranked.append(n)


# turn the constraints from the file into dictionaries
def receive_constraints(file):
    with open(file) as fp:
        constraints = {}
        row_num = col_num = goal = 0
        for line in fp:
            if line != DELIMITER:
                row_num, constraints, goal = add_constraint(row_num, line, constraints, goal, 'row')
            else:
                break
        for line in fp:
            col_num, constraints, goal = add_constraint(col_num,line,constraints,goal, 'col')
    return constraints, goal, row_num, col_num


# helper function for the constraint dictionary creation
def add_constraint(row_or_col_num, line, constraints, goal, row_or_col):
    row_or_col_num += 1
    list_int = list(map(int, line.split(' ')))
    constraints[row_or_col + str(row_or_col_num)] = np.asarray(list_int)
    goal += sum(x > 0 for x in list_int)
    return row_or_col_num, constraints, goal


# initialize population
def init_population(constraints):
    i = 0
    pop_holder = np.ndarray(POP_SIZE, dtype=object)
    while i < POP_SIZE:
        # for every row, set x 1's in random places in the row
        # where x is the number of 1's that are supposed to be in that row
        matrix = np.zeros((row_num, col_num))
        for j in range(row_num):
            row_constraints = constraints["row" + str(j + 1)]
            number_of_ones = np.sum(row_constraints)
            row = matrix[j]
            indices_to_change = np.random.choice(row.size, number_of_ones, replace=False)
            row.ravel()[indices_to_change] = 1
        pop_holder[i] = [matrix.astype(int), 0]
        i += 1
    return pop_holder


# give a fitness score to the given sequence (row or column) based on how many
# constraints it fills
def fit(sequence, sequence_constr, row_or_col):
    fill_black_counter = 0
    built_in_constrain = []
    length = row_num if row_or_col == 'col' else col_num
    for i in range(length):
        if sequence[i] == 0:
            if fill_black_counter > 0:
                built_in_constrain.append(fill_black_counter)
                fill_black_counter = 0
            continue
        else:
            fill_black_counter += 1
    if fill_black_counter > 0:
        built_in_constrain.append(fill_black_counter)

    built_in_constrain = np.asarray(built_in_constrain)
    sequence_constr = sequence_constr[sequence_constr != 0]

    # with order without zeros
    order_fitness_score = 0
    for n, m in zip(sequence_constr, built_in_constrain):
        if n == m:
            order_fitness_score += 1

    return order_fitness_score


# give a fitness function to every solution in the population
# based on the sum of the fitness scores it has for every row and column
def fitness_function(pop, constraints_dict, fitness_goal):
    global optimal_solution_found
    for s in range(len(pop)):
        solution = pop[s][0]
        fitness_score = 0
        for row in range(row_num):
            sub_fit_by_row = fit(solution[row], constraints_dict["row" + str(row + 1)], 'row')
            fitness_score += sub_fit_by_row
        for col in range(col_num):
            sub_fit_by_col = fit((np.transpose(solution))[col], constraints_dict["col" + str(col + 1)], 'col')
            fitness_score += sub_fit_by_col
        pop[s][1] = fitness_score
        if fitness_score == fitness_goal:
            optimal_solution_found = True
    return pop


def crossover(parent1, parent2):
    child1, child2 = np.zeros((row_num, col_num), dtype=int), np.zeros((row_num, col_num), dtype=int)
    # crossover rate is 99%
    rand = random.random()
    if rand < 1:
        for i in range(row_num):
            for j in range(col_num):
                get_bit_from_parent(i, j, child1, parent1, parent2)
                get_bit_from_parent(i, j, child2, parent1, parent2)
    else:
        child1, child2 = parent1, parent2
    return child1, child2


def get_bit_from_parent(i, j, child, parent1, parent2):
    # uniform crossover - every bit is chosen randomly from either one of the parents
    if random.random() < 0.5:
        child[i][j] = parent1[i][j]
    else:
        child[i][j] = parent2[i][j]


def mutate(child):
    # 4% of bits will be mutated
    mutation_percent = int((row_num*col_num)*0.04)
    # mutation rate is 5%
    if np.random.randint(20) == 1:
        indices_to_change = np.random.choice(child.size, mutation_percent, replace=False)
        for index in indices_to_change:
            child.ravel()[index] = 1 if child.ravel()[index] == 0 else 0
    return child


def mutation(child1, child2):
    child1, child2 = mutate(child1), mutate(child2)
    return child1, child2


# create new generation - elitism, selection, crossover and mutation
def new_gen(pre_pop):
    new_pop_holder = np.ndarray(pre_pop.size, dtype=object)
    sorted_prev_pop = np.asarray(sorted(pre_pop, key=lambda t: t[1]), dtype=object)
    counter = 0
    # first we are going to preform elitism, by inserting top k solution to new_pop_holder
    number_of_elitism = int(pre_pop.size / (row_num+col_num))
    while number_of_elitism != 0:
        new_pop_holder[counter] = sorted_prev_pop[len(sorted_prev_pop) - math.ceil(number_of_elitism / 2)]
        sorted_prev_pop[len(sorted_prev_pop) - math.ceil(number_of_elitism / 2)] = sorted_prev_pop[
            len(sorted_prev_pop) - number_of_elitism]
        number_of_elitism -= 1
        counter += 1

    # create the rest of the new generation
    while counter != pre_pop.size:
        # pick randomly 2 solutions for crossover and mutation
        random_indexes1, random_indexes2 = ranked_by_fitness(pre_pop.size)
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
    return new_pop_holder, sorted_prev_pop[len(sorted_prev_pop) - 1][1], sorted_prev_pop[len(sorted_prev_pop) - 1]


def ranked_by_fitness(pop_size):
    if pop_size == NICHE_SIZE:
        ranked_fit = niche_ranked
    else:
        ranked_fit = ranked
    place = np.random.choice(len(ranked_fit), 2, replace=False)
    return ranked_fit[place[0]], ranked_fit[place[1]]


def create_new_gen(population, constraints, finish_line):
    population = fitness_function(population, constraints, finish_line)
    population, best_fit, fittest_sol = new_gen(population)
    return population, best_fit, fittest_sol



def start(file):
    global row_num, col_num
    constraints, finish_line, row_num, col_num = receive_constraints(file)
    population = init_population(constraints)
    generation = 0
    divide_to_niches = False
    np.random.shuffle(population)
    pop_all_in = []
    best_in_niche = [0 for i in range(NICHE_NUM+1)]
    fittest_sols = [0 for i in range(NICHE_NUM+1)]
    niches = [[] for i in range(NICHE_NUM)]

    print("The algorithm will run a maximum of " + str(TOTAL_GENS) + " generations")
    start = time()
    # run for TOTAL_GENS generations, or until optimal solution is found
    while generation < TOTAL_GENS:
        # divide the population into niches
        if generation == 0:
            for i in range(NICHE_NUM):
                niches[i] = population[NICHE_SIZE*i:NICHE_SIZE*(i+1)]
        elif divide_to_niches:
            for j in range(NICHE_NUM):
                niches[j] = pop_all_in[NICHE_SIZE * j:NICHE_SIZE * (j + 1)]
            divide_to_niches = False

        # create new generation and for every niche save the best solution
        for k in range(NICHE_NUM):
            niches[k], best_in_niche[k], fittest_sols[k] = create_new_gen(niches[k], constraints, finish_line)
        cur_fittest = np.argmax(np.asarray(best_in_niche))

        print("best solution: {:.2%}".format(best_in_niche[cur_fittest] / finish_line))
        generation += 1
        print("generation: " + str(generation) + "\n")

        if optimal_solution_found:
            break

        # combine all niches into one population and create 2 new generations
        if generation % 20 == 0 and generation != TOTAL_GENS:
            pop_all_in = np.concatenate(niches)
            for k in range(2):
                pop_all_in, best_in_niche[NICHE_NUM], fittest_sols[NICHE_NUM] = create_new_gen(pop_all_in, constraints, finish_line)
                cur_fittest = np.argmax(np.asarray(best_in_niche))
                print("best solution: {:.3%}".format(best_in_niche[cur_fittest] / finish_line))
                generation += 1
                print("generation: " + str(generation) + "\n")
            np.random.shuffle(pop_all_in)
            divide_to_niches = True

    end = time()
    print("solution found in {:.3} seconds".format(str(end - start)))
    Plot.plot_grid(fittest_sols[cur_fittest][0], file)
