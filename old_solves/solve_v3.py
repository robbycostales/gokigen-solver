from __future__ import division
from __future__ import print_function

# Author: Robby Costales
# Date: 2018-04-15
# Language: Python 2

# Purpose: run tests on solving algorithm

# VERSION: networkx implementation

# common
import glob
import time
import copy
import random
import numpy as np
import cProfile
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.cycles import find_cycle
# local
import read

class Solver:
    def __init__(self, filename=None):
        # initialize attributes
        self.is_solved = False
        self.puzzle = None
        self.solution = None
        # from params
        self.filename = filename
        # read puzzle if provided filename
        if filename:
            self.read_puzzle(filename)

        pass

    def mn_to_i(self, m, n):
        return m*(self.rows-1) + n

    def i_to_mn(self, i):
        return i // (self.cols -1), i % (self.cols -1)

    def yx_to_i(self, y, x):
        return y*(self.rows) + x

    def i_to_yx(self, i):
        return i // (self.cols), i % (self.cols)

    def verify_solution(self):
        if self.solution == self.state:
            return True
        else:
            return False
        pass

    def print_puzzle(self):
        print("")
        print(self.filename)
        print("rows:{}".format(self.rows))
        print("cols:{}".format(self.cols))
        print(np.matrix(self.puzzle))
        print(np.matrix(self.solution))

    def print_state(self):
        print(np.matrix(self.state))

    def read_puzzle(self, filename):
        """
        Reads puzzle given filename
        """
        # call function from module `read`
        self.puzzle, self.solution = read.read_puzzle(filename)
        self.is_solved = False
        self.rows = len(self.puzzle)
        self.cols = len(self.puzzle[0])
        self.size = self.rows *self.cols
        self.maxi = (self.rows-1) * (self.rows-1)

    def solve(self, maxtime=60):
        self.maxtime = maxtime
        # initalize state
        self.state = [["-" for j in range(self.cols-1)] for i in range(self.rows-1)]
        self.counts = [[0 for j in range(self.cols)] for i in range(self.rows)]
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.i_to_yx(i) for i in range(self.size))
        # backtracking, initial call
        self.solve_start = time.time()
        self.backtrack(0)
        self.solve_end = time.time()
        self.solve_time = self.solve_end - self.solve_start

    def place(self, m, n, slash):
        self.state[m][n] = slash
        if slash == "\\":
            self.counts[m][n] += 1
            self.counts[m+1][n+1] += 1
            self.graph.add_edge((m,n),(m+1, n+1))
        else:
            self.counts[m+1][n] += 1
            self.counts[m][n+1] += 1
            self.graph.add_edge((m+1,n),(m, n+1))

    def remove(self, m, n, slash):
        self.state[m][n] = "-"
        if slash == "\\":
            self.counts[m][n] -= 1
            self.counts[m+1][n+1] -= 1
            self.graph.remove_edge((m,n),(m+1, n+1))
        else:
            self.counts[m+1][n] -= 1
            self.counts[m][n+1] -= 1
            self.graph.remove_edge((m+1,n),(m, n+1))

    def backtrack(self, nexti):
        """
        backtracking algorithm, takes first call and all subsequent
        nexti - number that will determine m and n
        """
        # time and if solved
        if time.time() > self.solve_start + self.maxtime or self.is_solved == True:
            return

        # set cur row and col
        m,n = self.i_to_mn(nexti)

        choices = ["/", "\\"]

        for slash in choices:
            # if we already found solution, return
            if self.is_solved == True:
                return

            # place the slash, and check consequences
            self.place(m, n, slash)

            if self.check_neighbors(nexti) and self.check_cycles(nexti):
                # check that the neighboring numbers can be satistfied after this placement
                # check that there are no cycles

                # we want to keep placement, and make next call
                if nexti == (self.rows-1)*(self.cols-1) -1:
                    # if we are at the end, then wE DONE
                    self.is_solved = True
                    return
                else:
                    # next calls
                    self.backtrack(nexti+1)
                    if self.is_solved == True:
                        return
                    else:
                        # reset current placement
                        self.remove(m, n, slash)
                    pass # end of else (not final)
                pass # end of 'if <checks>'
            else:
                # reset current placement
                self.remove(m, n, slash)

            pass # end of 'for slash...' loop
        pass # end of function


    def check_cycles(self, i):
        """
        Check cycles at current index (i)
        Returns boolean - True if no cycles, False if cycles
        """
        m,n = self.i_to_mn(i)

        init_slash = self.state[m][n]

        # if we put this character down, we couldn't have created a cycle
        # RULE 1
        if m == 0 or n == 0:
            return True
        # RULE 2
        elif init_slash == "\\":
            return True
        # RULE 3
        elif init_slash == "/" and self.state[m][n-1] != "\\":
            return True
        # RULE 4
        elif init_slash == "/": # we know self.state[m][n-1] == "\\"
            if n == self.cols - 2 and self.state[m-1][n] == "/":
                return True
            elif n != self.cols -2 and self.state[m-1][n] == "/" and self.state[m-1][n+1] == "\\":
                return True


        if init_slash == "/":
            try:
                find_cycle(self.graph, (m+1, n))
            except:
                return True
            return False
        else:
            try:
                find_cycle(self.graph, (m, n+1))
            except:
                return True
            return False



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
                if 0 <= y < self.rows-1 and 0 <= x < self.cols-1:
                    new_slash = self.state[y][x]
                    # specific slash rule for number:
                    if new_slash == slash and slash == "\\":
                        new_frontier.append((new_slash, y, x))
                # 2
                y = cur_m - 1
                x = cur_n
                if 0 <= y < self.rows-1 and 0 <= x < self.cols-1:
                    new_slash = self.state[y][x]
                    # specific slash rule for number:
                    if new_slash != slash and new_slash != "-":
                        new_frontier.append((new_slash, y, x))
                # 3
                y = cur_m - 1
                x = cur_n + 1
                if 0 <= y < self.rows-1 and 0 <= x < self.cols-1:
                    new_slash = self.state[y][x]
                    # specific slash rule for number:
                    if new_slash == slash and slash == "/":
                        new_frontier.append((new_slash, y, x))
                # 4
                y = cur_m
                x = cur_n + 1
                if 0 <= y < self.rows-1 and 0 <= x < self.cols-1:
                    new_slash = self.state[y][x]
                    # specific slash rule for number:
                    if new_slash != slash and new_slash != "-":
                        new_frontier.append((new_slash, y, x))
                # 5
                y = cur_m + 1
                x = cur_n + 1
                if 0 <= y < self.rows-1 and 0 <= x < self.cols-1:
                    new_slash = self.state[y][x]
                    # specific slash rule for number:
                    if new_slash == slash and slash == "\\":
                        new_frontier.append((new_slash, y, x))
                # 6
                y = cur_m + 1
                x = cur_n
                if 0 <= y < self.rows-1 and 0 <= x < self.cols-1:
                    new_slash = self.state[y][x]
                    # specific slash rule for number:
                    if new_slash != slash and new_slash != "-":
                        new_frontier.append((new_slash, y, x))
                # 7
                y = cur_m + 1
                x = cur_n - 1
                if 0 <= y < self.rows-1 and 0 <= x < self.cols-1:
                    new_slash = self.state[y][x]
                    # specific slash rule for number:
                    if new_slash == slash and new_slash == "/":
                        new_frontier.append((new_slash, y, x))
                # 8
                y = cur_m
                x = cur_n - 1
                if 0 <= y < self.rows-1 and 0 <= x < self.cols-1:
                    new_slash = self.state[y][x]
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

            pass # end of while(frontier) loop

        return True


    def check_neighbors(self, nexti):
        """
        checks if current placement ruins the chances of any neighbors reaching their intersection goal
        nexti - current index we are working on in state (and the slash has already been placed when this function is called)
        returns Boolean - True if no violation, False if violation
        """
        # set row and col
        m,n = self.i_to_mn(nexti)

        # upper left, upper right, lower left, lower right
        UL = self.puzzle[m][n]
        UR = self.puzzle[m][n+1]
        LL = self.puzzle[m+1][n]
        LR = self.puzzle[m+1][n+1]

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
            if 0 <= y-1 < self.rows-1 and 0 <= x-1 < self.cols-1:
                val = self.state[y-1][x-1]
                if val == "\\":
                    connected += 1
                elif val == "-":
                    unknown +=1

            # check upper right OF corner
            if 0 <= y-1 < self.rows-1 and 0 <= x < self.cols-1:
                val = self.state[y-1][x]
                if val == "/":
                    connected += 1
                elif val == "-":
                    unknown +=1

            # check lower left OF corner
            if 0 <= y < self.rows-1 and 0 <= x-1 < self.cols-1:
                val = self.state[y][x-1]
                if val == "/":
                    connected += 1
                elif val == "-":
                    unknown +=1

            # check lower right OF corner
            if 0 <= y < self.rows-1 and 0 <= x < self.cols-1:
                val = self.state[y][x]
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


if __name__ == "__main__":
    PRINT_SIZES = False


    total_solved = 0
    total_unsolved = 0

    # get all filenames sorted from puzzles folder
    filenames = glob.glob("puzzles/*.txt")
    filenames.sort()

    solve_times = []

    sizes = Counter()

    run_start = time.time()

    for i in range(len(filenames)):
        # if i != 9:
        #     continue

        solver = Solver(filenames[i])

        if PRINT_SIZES:
            sizes[solver.size] += 1
            continue

        try:
            perc = (total_solved / (total_solved + total_unsolved))*100
        except:
            perc = 100
        print("running test {} / {}   ({} / {} found so far... {}%)".format(str(i+1), str(len(filenames)), str(total_solved), str(total_solved + total_unsolved), perc))

        if  not (121 <= solver.size <= 121):
            print("skipping size {}...".format(solver.size))
            continue

        solver.print_puzzle()
        # solver.solve(60)
        cProfile.run('solver.solve(60)')
        solver.print_state()


        if solver.verify_solution():
            print("SOLVED")
            total_solved += 1
            solve_times.append(solver.solve_time)
        else:
            print("unsolved...")
            total_unsolved += 1

        print(solver.solve_time)
        print("")

    if PRINT_SIZES:
        print(sizes.most_common())

    else:
        print("total solved: %d" % total_solved)
        print("total unsolved: %d" % total_unsolved)
        if len(solve_times):
            print("avg solve time: {:.4f}".format(sum(solve_times)/len(solve_times)))
    print("run time: {:.2f}".format(time.time() - run_start))
    # plt.hist(solve_times)
    # plt.show()
