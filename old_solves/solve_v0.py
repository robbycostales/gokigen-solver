from __future__ import division
from __future__ import print_function

# Author: Robby Costales
# Date: 2018-04-15
# Language: Python 2

# Purpose: run tests on solving algorithm

# IMPORT STATEMENTS
import glob
import time
import copy

import random
import numpy as np

import cProfile

# GLOBAL VARIABLES
global isSolved
global finishTime
global puzzle, solution, rows, cols



def verifySolution():
    """
    Compares the known solution with the generated / solved solution found

    Args:
        (none because we use global vars)
    Returns:
        Boolean: True if correct solution, False if not
    """
    global state, solution

    if state == solution:
        return True
    else:
        return False


def readPuzzle(fileName):
    """
    Function for getting problem and solution information from each puzzle in the /puzzles/ directory

    Args:
        fileName : name of file
    Returns:
        problem, solution -- in form of nested lists
    """
    f = open(fileName, "r")
    x = ""

    # skip the beginning part
    while "problem" not in x:
        x = f.readline()

    # read problem
    problem = []
    while "solution" not in x:
        x = f.readline()
        problem.append(list(x.split(" ")))
    # remove last element which contains solution
    problem.pop()

    # read solution
    solution = []
    while "moves" not in x:
        x = f.readline()
        solution.append(list(x.split(" ")))
    # remove last again
    solution.pop()

    # fix problem values
    for i in range(len(problem)):
        for j in range(len(problem[i])):
            # get rid of any \n
            problem[i][j] = problem[i][j].replace("\n", "")

            if problem[i][j] == "-":
                problem[i][j] = -1
            else:
                problem[i][j] = int(problem[i][j])

    # fix solution values
    for i in range(len(solution)):
        for j in range(len(solution[i])):
            # get rid of any \n
            solution[i][j] = solution[i][j].replace("\n", "")

    return problem, solution


def check_neighbors(nexti):
    """
    checks if current placement ruins the chances of any neighbors reaching their intersection goal

    Args:
        nexti - current index we are working on in state (and the slash has already been placed when this function is called)
    Returns:
        Boolean - True if no violation, False if violation
    """
    global puzzle
    global rows, cols
    global state

    # set row
    m = nexti // (cols -1)
    # set col
    n = nexti % (cols -1)

    # upper left, upper right, lower left, lower right
    UL = puzzle[m][n]
    UR = puzzle[m][n+1]
    LL = puzzle[m+1][n]
    LR = puzzle[m+1][n+1]

    for corner, pos in [(UL, (m,n)), (UR, (m,n+1)), (LL, (m+1,n)), (LR, (m+1,n+1))]:
        if corner == -1:
            # we don't have to worry about the -1's
            continue

        y = pos[0]
        x = pos[1]

        # init values
        connected = 0
        unknown = 0

        # check upper left OF corner
        if 0 <= y-1 < rows-1 and 0 <= x-1 < cols-1:
            val = state[y-1][x-1]
            if val == "\\":
                connected += 1
            elif val == "-":
                unknown +=1

        # check upper right OF corner
        if 0 <= y-1 < rows-1 and 0 <= x < cols-1:
            val = state[y-1][x]
            if val == "/":
                connected += 1
            elif val == "-":
                unknown +=1

        # check lower left OF corner
        if 0 <= y < rows-1 and 0 <= x-1 < cols-1:
            val = state[y][x-1]
            if val == "/":
                connected += 1
            elif val == "-":
                unknown +=1

        # check lower right OF corner
        if 0 <= y < rows-1 and 0 <= x < cols-1:
            val = state[y][x]
            if val == "\\":
                connected += 1
            elif val == "-":
                unknown +=1


        # check if corner value can be met
        if connected <= corner <= connected + unknown:
            continue
        else:
            return False

    return True


def check_cycles(p=False):
    """
    checks if current placement creates any cycles

    Args:
        (none, uses global variables)
    Returns:
        Boolean - True if no cycles, False if cycles
    """

    global rows, cols
    global state


    # all found points
    all_found_coords = []

    # for each cell
    for i in range((cols-1)*(rows-1)-1):
        # set row
        m = i // (cols-1)
        # set col
        n = i % (cols-1)
        # current point

        init_slash = state[m][n]
        if init_slash == "-":
            continue

        # if path already explored, move on
        if (m,n) in all_found_coords:
            continue

        # each point in form (slash, m, n, prev_m, prev_n)
        frontier = [(init_slash, m, n)]
        # initialize path's found points
        cur_found_coords = [(frontier[0][1], frontier[0][2])]
        # new previous frontier at beginning
        prev_frontier_coords = []
        while len(frontier) != 0:

            # for new batch
            new_frontier = []
            # for each value in the frontier
            for val in frontier:
                slash = val[0]
                cur_m = val[1]
                cur_n = val[2]
                # find all neighbors that are not the previous

                # 1
                y = cur_m - 1
                x = cur_n - 1
                if 0 <= y < rows-1 and 0 <= x < cols-1:
                    new_slash = state[y][x]
                    # specific slash rule for number:
                    if new_slash == slash and slash == "\\":
                        new_frontier.append((new_slash, y, x))
                # 2
                y = cur_m - 1
                x = cur_n
                if 0 <= y < rows-1 and 0 <= x < cols-1:
                    new_slash = state[y][x]
                    # specific slash rule for number:
                    if new_slash != slash and new_slash != "-":
                        new_frontier.append((new_slash, y, x))
                # 3
                y = cur_m - 1
                x = cur_n + 1
                if 0 <= y < rows-1 and 0 <= x < cols-1:
                    new_slash = state[y][x]
                    # specific slash rule for number:
                    if new_slash == slash and slash == "/":
                        new_frontier.append((new_slash, y, x))
                # 4
                y = cur_m
                x = cur_n + 1
                if 0 <= y < rows-1 and 0 <= x < cols-1:
                    new_slash = state[y][x]
                    # specific slash rule for number:
                    if new_slash != slash and new_slash != "-":
                        new_frontier.append((new_slash, y, x))
                # 5
                y = cur_m + 1
                x = cur_n + 1
                if 0 <= y < rows-1 and 0 <= x < cols-1:
                    new_slash = state[y][x]
                    # specific slash rule for number:
                    if new_slash == slash and slash == "\\":
                        new_frontier.append((new_slash, y, x))
                # 6
                y = cur_m + 1
                x = cur_n
                if 0 <= y < rows-1 and 0 <= x < cols-1:
                    new_slash = state[y][x]
                    # specific slash rule for number:
                    if new_slash != slash and new_slash != "-":
                        new_frontier.append((new_slash, y, x))
                # 7
                y = cur_m + 1
                x = cur_n - 1
                if 0 <= y < rows-1 and 0 <= x < cols-1:
                    new_slash = state[y][x]
                    # specific slash rule for number:
                    if new_slash == slash and new_slash == "/":
                        new_frontier.append((new_slash, y, x))
                # 8
                y = cur_m
                x = cur_n - 1
                if 0 <= y < rows-1 and 0 <= x < cols-1:
                    new_slash = state[y][x]
                    # specific slash rule for number:
                    if new_slash != slash and slash != "-":
                        new_frontier.append((new_slash, y, x))

            # get rid of ones from previous generation
            news = []
            for new in new_frontier:
                if (new[1], new[2]) not in prev_frontier_coords and new not in frontier:
                    news.append(new)
            new_frontier = news

            # check that there are no duplicates found
            to_set = [(i[1], i[2]) for i in new_frontier]
            if len(to_set) != len(set(to_set)):
                return False

            # check that none of the new_frontier have been found before in path
            for new in new_frontier:
                if (new[1], new[2]) in cur_found_coords:
                    return False

            # update frontier
            prev_frontier_coords = [(i[1], i[2]) for i in frontier]
            cur_found_coords = [(i[1], i[2]) for i in new_frontier]
            frontier = new_frontier

        all_found_coords += cur_found_coords
    return True


def backtrack(nexti):
    """
    backtracking algorithm, takes first call and all subsequent

    Args:
        nexti - number that will determine m and n

    Returns:
        not sure yet
    """
    global isSolved
    global finishTime
    global puzzle, solution, rows, cols
    global state
    global SOLS

    # set row
    m = nexti // (cols -1)
    # set col
    n = nexti % (cols -1)

    # check if we have run out of time
    if time.time() > finishTime or isSolved == True:
        return

    # initialization
    if nexti == 0:
        # the -1 is because the slashes are in between the numbers in the puzzle
        state = [["-" for j in range(cols-1)] for i in range(rows-1)]

    for slash in ["/", "\\"]:
        # if we already found solution, return
        if isSolved == True:
            return

        # place the slash, and check consequences
        state[m][n] = slash

        if check_neighbors(nexti): # and check_cycles():
            # check that the nieghboring numbers can be satistfied after this placement
            # check that there are no cycles

            # we want to keep placement, and make next call
            if nexti == (rows-1)*(cols-1) -1:
                # if we are at the end, then wE DONE
                if check_cycles():
                    isSolved = True
                    return
            else:
                # next calls
                backtrack(nexti+1)
                if isSolved == True:
                    return
                else:
                    # reset current placement
                    state[m][n] = "-"

        else:
            # reset current placement
            state[m][n] = "-"

def solve():
    global state
    global SOLS


    backtrack(0)

    # for sol in SOLS:
    #     if time.time() > finishTime:
    #         return
    #     state = sol
    #     if check_cycles():
    #         return

if __name__ == "__main__":
    global isSolved
    global finishTime
    global puzzle, solution, rows, cols
    global state
    global SOLS

    totalSolved = 0
    totalUnsolved = 0

    fileNames = glob.glob("puzzles/*.txt")
    fileNames.sort()

    for i in range(len(fileNames)):
        if i == 0:
            perc = 100
        else:
            perc = (totalSolved / i)*100
        print("running test {} / {}   ({} / {} found so far... {}%)".format(str(i+1), str(len(fileNames)), str(totalSolved), str(i), perc))


        puzzle, solution = readPuzzle(fileNames[i])
        print(np.matrix(puzzle))
        print(np.matrix(solution))
        rows = len(puzzle)
        cols = len(puzzle[0])

        print(fileNames[i])
        print("rows:{}".format(rows))
        print("cols:{}".format(cols))

        startTime = time.time()
        duration = 60
        finishTime = startTime + duration
        isSolved = False

        SOLS = []

        solve()

        endTime = time.time()

        if verifySolution():
            print("SOLVED")
            totalSolved += 1
        else:
            print("unsolved...")
            totalUnsolved += 1

        print(endTime - startTime)
        print("")
    print("total solved: %d" % totalSolved)
    print("total unsolved: %d" % totalUnsolved)
