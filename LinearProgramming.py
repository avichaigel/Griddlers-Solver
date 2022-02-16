"""
Solve the Griddler game using Linear Programming
"""
from __future__ import print_function
import numpy as np
from time import time
from gurobipy import Model, GRB, quicksum
import io
import Plot

CASE_WHITE, CASE_BLACK, CASE_BLANK = -1, 1, 0
MODEL_NAME = "lp_griddler_solver"
DELIMITER = "#\n"
FILE = "16.txt"

def linear_programming(line_constraints, column_constraints, grid):
    global MODEL_NAME

    # Declaration
    N, M, = len(line_constraints), len(column_constraints)
    model = Model(MODEL_NAME)
    model.setParam('OutputFlag', False)

    # Calculation of possible boxes (boxes that can be blacked out)
    authorized_Y = possible_cases(line_constraints, N, M)
    authorized_Z = possible_cases(column_constraints, M, N)

    # Variables type 1 X i,j
    lx = np.array([[model.addVar(vtype=GRB.BINARY) for j in range(M)] for i in range(N)])

    # Variables type 2 Y i,j,t
    ly = np.array([[[model.addVar(vtype=GRB.BINARY) if j in authorized_Y[i][t] else None
                        for t in range(len(line_constraints[i]))] for j in range(M)] for i in range(N)],dtype=object)

    # Variables type 2 Z i,j,t
    lz = np.array([[[model.addVar(vtype=GRB.BINARY) if i in authorized_Z[j][t] else None
                        for t in range(len(column_constraints[j]))] for i in range(N)] for j in range(M)],dtype=object)

    # Add constraints to model
    # line constraints
    for i in range(N):
        for t in range(len(line_constraints[i])):
            l1 = [ly[i, k][t] for k in range(M) if ly[i, k][t]]
            if len(l1) > 0:
                model.addConstr(quicksum(key1 for key1 in l1), GRB.EQUAL, 1)

    for i in range(N):
        for j in range(M):
            for t in range(len(line_constraints[i])):
                if ly[i, j][t]:
                    l1 = []
                    for t1 in range(t + 1, len(line_constraints[i])):
                        l1 += [ly[i, k][t1] for k in range(j + line_constraints[i][t] + 1) if k < M and ly[i, k][t1]]
                    if len(l1) > 0:
                        model.addConstr(len(l1) * ly[i, j][t] + quicksum(keyy for keyy in l1) <= len(l1))
                    l1 = [lx[i, k] for k in range(j, j + line_constraints[i][t]) if k < M]
                    if len(l1) > 0:
                        model.addConstr(line_constraints[i][t] * ly[i, j][t] <= quicksum(keyx for keyx in l1))
            for t in range(len(column_constraints[j])):
                if lz[j, i][t]:
                    l1 = []
                    for t1 in range(t + 1, len(column_constraints[j])):
                        l1 += [lz[j, k][t1] for k in range(i + column_constraints[j][t] + 1) if k < N and lz[j, k][t1]]
                    if len(l1) > 0:
                        model.addConstr(len(l1) * lz[j, i][t] + quicksum(keyz for keyz in l1) <= len(l1))
                    l1 = [lx[k, j] for k in range(i, i + column_constraints[j][t]) if k < N]
                    if len(l1) > 0:
                        model.addConstr(column_constraints[j][t] * lz[j, i][t] <= quicksum(keyx for keyx in l1))

    # column constraints
    for j in range(M):
        for t in range(len(column_constraints[j])):
            l1 = [lz[j, i][t] for i in range(N) if lz[j, i][t]]
            if len(l1) > 0:
                model.addConstr(quicksum(key1 for key1 in l1), GRB.EQUAL, 1)

    model.setObjective(sum(sum(lx)), GRB.MINIMIZE)
    model.update()
    t1 = time()
    model.optimize()
    t2 = time()

    # build the final grid
    for i in range(N):
        for j in range(M):
            if grid[i, j] == CASE_BLANK:
                values = lx[i, j].x
                if values < 0 or (values >= 0 and values < .5):
                    grid[i, j] = CASE_WHITE
                else:
                    grid[i, j] = CASE_BLACK

    return grid, t2 - t1

def possible_cases(sequence, N, M):
    """
    Returns the list of possible boxes for each block of the N lines over M columns.
    """
    possible_cases = []
    for i in range(N):
        L = []
        l = sequence[i]
        for t in range(len(l)):
            forbidden_cases = []
            o = 0
            # The front blocks
            for t1 in range(0, t):
                for h in range(l[t1]):
                    forbidden_cases += [o]
                    o += 1
                # Blank cases
                forbidden_cases += [o]
                o += 1
            # cross the line
            for o in range(M - 1, 0, -1):
                if o + l[t] > M:
                    forbidden_cases += [o]
            o = M - 1
            # The last blocks
            for t1 in range(len(l) - 1, t, -1):
                for h in range(l[t1]):
                    forbidden_cases += [o]
                    o -= 1
                # White cases
                forbidden_cases += [o]
                o -= 1
            # Do not confuse between the current block and the blocks after
            for t1 in range(l[t], 1, -1):
                forbidden_cases += [o]
                o -= 1
            L.append([case for case in range(M) if case not in forbidden_cases])
        possible_cases.append(L)
    return possible_cases

def parse_instance(file_path):
    global DELIMITER

    lines, columns = [], []
    with io.open(file_path, encoding="utf8") as file:
        for line in file:
            if line != DELIMITER:
                lines.append(parse_line(line))
            else:
                break
        for line in file:
            columns.append(parse_line(line))
    return lines, columns

def parse_line(line):
    """ Parses a line containing integers separated by a space, or an
     empty line, then returns a list of integers, or an empty list.
    """
    return list(map(int, line.split()))

def solve(line_constraints, column_constraints, grid):
    grid, execution_time = linear_programming(line_constraints, column_constraints, grid)
    return grid, execution_time

if __name__ == '__main__':
    line_constraints, column_constraints = parse_instance(FILE)
    grid = np.full((len(line_constraints), len(column_constraints)), CASE_BLANK)
    grid, execution_time = solve(line_constraints, column_constraints, grid)
    print("solution found in " + str(execution_time) + " seconds")
    Plot.plot_grid(grid, FILE)
