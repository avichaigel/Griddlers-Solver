import GeneticAlgorithm
import LinearProgramming
import matplotlib.pyplot as plt


stop = False
print("Welcome to Griddler solver!")
while not stop:
    print("Please choose algorithm to solver the Griddler puzzle with:")
    print("1. Solve with Linear Programming algorithm")
    print("2. Solve with Genetic Programming algorithm")
    chosen_alg = input()
    while chosen_alg != '1' and chosen_alg != '2':
        chosen_alg = input("Wrong key pressed. Please choose 1 or 2")
    if chosen_alg == '1':
        solver = LinearProgramming
        solver_name = 'Linear Programming'
    else:  # chosen_alg == '2'
        solver = GeneticAlgorithm
        solver_name = 'Genetic Algorithm'
    runttime_lst = []
    best_sol_lst = []
    rt_avg=0
    best_avg = 0
    file = ''
    if chosen_alg == '1':
        i = 1
        while i <=3:
            if i == 1:
                file = 'easy-key.txt'
                level_name = 'easy'
            elif i == 2:
                file = 'medium-drops.txt'
                level_name = 'medium'
            else:  # level == '3'
                file = 'hard-bells.txt'
                level_name = 'hard'
            print("Solving " + level_name + " level Griddler with " + solver_name)
            runttime = solver.start(file)
            runttime_lst.append(runttime)
            i += 1
        tick_lable = ['easy','medium','hard']
        left = [1, 2, 3]
        plt.xlabel('Puzzle Level')
        plt.ylabel('Runtime')
        plt.bar(left, runttime_lst, tick_label=tick_lable,
                width=0.4, color=['blue'])
        plt.show()
    else:
        i =1
        while i <= 3:
            if i == 1:
                file = 'easy-key.txt'
                level_name = 'easy'
            elif i == 2:
                file = 'medium-drops.txt'
                level_name = 'medium'
            else:  # level == '3'
                file = 'hard-bells.txt'
                level_name = 'hard'
            j = 1
            while j <= 3:
                print("Solving " + level_name + " level Griddler with " + solver_name)
                runttime, best_sol = solver.start(file)
                rt_avg += runttime
                best_avg += best_sol
                print(j)
                j += 1
            rt_avg = rt_avg / 3
            best_avg = best_avg / 3
            print(best_avg)
            runttime_lst.append(rt_avg)
            best_sol_lst.append(best_avg)
            best_avg = 0
            rt_avg = 0
            i += 1
            print(i)
        print(runttime_lst)
        print(best_sol_lst)
        tick_lable = ['easy', 'medium', 'hard']
        left = [1, 2, 3]
        plt.xlabel('Puzzle Level')
        plt.ylabel('Average Runtime')
        plt.bar(left, runttime_lst, tick_label=tick_lable,
                width=0.4, color=['blue'])
        plt.show()

        tick_lable = ['easy', 'medium', 'hard']
        left = [1, 2, 3]
        plt.xlabel('Puzzle Level')
        plt.ylabel('Average of best solution')
        plt.bar(left, best_sol_lst, tick_label=tick_lable,
                width=0.4, color=['blue'])
        plt.show()
    stop_choice = input("\nTo solve another puzzle enter 1, to exit enter 0")
    stop = True if stop_choice == '0' else False
