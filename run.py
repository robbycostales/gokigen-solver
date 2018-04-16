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
import numpy

# GLOBAL VARIABLES
global isSolved
global finishTime
global puzzle, solution, rows, cols


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



def backtrack(m, n):
    """
    backtracking algorithm, takes first call and all subsequent

    Args:
        m : current row
        n : current column

    Returns:
        not sure yet
    """
    global isSolved
    global finishTime
    global puzzle, solution, rows, cols

    if m == 0 and n == 0:
        state =



if __name__ == "__main__":
    global isSolved
    global finishTime
    global puzzle, solution, rows, cols

    totalSolved = 0
    totalUnsolved = 0

    fileNames = glob.glob("puzzles/*.txt")
    fileNames.sort()

    for fileName in fileNames:
        puzzle, solution = readPuzzle(fileName)
        rows = len(puzzle)
        cols = len(puzzle[0])

        print(fileName)

        startTime = time.time()
        duration = 100
        finishTime = startTime + duration
        isSolved = False

        backtrack(0, 0)

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
