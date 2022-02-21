import GeneticAlgorithm
import LinearProgramming

stop = False
print("Welcome to Griddler solver!")
while not stop:
    print("Please choose algorithm to solver the Griddler puzzle with:")
    print("1. Solve with Linear Programming algorithm")
    print("2. Solve with Genetic Programming algorithm")
    chosen_alg = input()
    while chosen_alg != '1' and chosen_alg != '2':
        chosen_alg = input("Wrong key pressed. Please choose 1 or 2")

    print("Please choose level of Griddler puzzle:\n1. Easy\n2. Medium\n3. Hard")
    level = input()
    while level != '1' and level != '2' and level != '3':
        level = input("Wrong key pressed. Please choose 1,2 or 3")

    if chosen_alg == '1':
        solver = LinearProgramming
        solver_name = 'Linear Programming'
    else:  # chosen_alg == '2'
        solver = GeneticAlgorithm
        solver_name = 'Genetic Algorithm'

    file = ''
    if level == '1':
        file = 'easy-key.txt'
        level_name = 'easy'
    elif level == '2':
        file = 'medium-drops.txt'
        level_name = 'medium'
    else:  # level == '3'
        file = 'hard-bells.txt'
        level_name = 'hard'

    print("Solving " + level_name + " level Griddler with " + solver_name)
    solver.start(file)

    stop_choice = input("\nTo solve another puzzle enter 1, to exit enter 0")
    stop = True if stop_choice == '0' else False
