from __future__ import division
from __future__ import print_function

# Author: Robby Costales
# Date: 2019-04-01
# Language: Python 3

# Purpose: run tests on solving algorithm
# before radical changes to check cycles

# common
import glob
import time
import copy
import random
import numpy as np
import cProfile
from collections import Counter, defaultdict
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
        # backtracking, initial call
        self.solve_start = time.time()
        self.backtrack(0)
        self.solve_end = time.time()
        self.solve_time = self.solve_end - self.solve_start

    def place(self, m, n, slash):
        self.state[m][n] = slash

    def remove(self, m, n, slash):
        self.state[m][n] = "-"

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

        # each point in form (slash, m, n, prev_m, prev_n)
        frontier = [(init_slash, m, n)]
        # initialize path's found points
        cur_found_coords = {(m, n)}
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
                if slash == "\\" and 0 <= y < self.rows-1 and 0 <= x < self.cols-1:
                    new_slash = self.state[y][x]
                    # specific slash rule for number:
                    if new_slash == slash:
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
                if slash == "/" and 0 <= y < self.rows-1 and 0 <= x < self.cols-1:
                    new_slash = self.state[y][x]
                    # specific slash rule for number:
                    if new_slash == slash:
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
                if slash == "\\" and 0 <= y < self.rows-1 and 0 <= x < self.cols-1:
                    new_slash = self.state[y][x]
                    # specific slash rule for number:
                    if new_slash == slash:
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
                if slash != "-" and 0 <= y < self.rows-1 and 0 <= x < self.cols-1:
                    new_slash = self.state[y][x]
                    # specific slash rule for number:
                    if new_slash != slash:
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
            cur_found_coords = set()
            for i in new_frontier:
                cur_found_coords.add((i[1], i[2]))
            # cur_found_coords = [(i[1], i[2]) for i in new_frontier]
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


def do_bargraph(data, labels):


    x = data
    x_pos = [i for i, _ in enumerate(x)]

    plt.bar(x_pos, data, color='green')
    plt.xlabel("puzzle size")
    plt.ylabel("percentage solved")
    plt.title("Percentage solved per puzzle size")

    plt.xticks(x_pos, labels)

    plt.show()

def do_boxplot(data, labels):
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    flierprops = dict(marker='+', markerfacecolor='None', markersize=8,
                  linestyle='none', markeredgecolor='k')

    bp = ax.boxplot(data, patch_artist=True, flierprops=flierprops)
    ax.set_xticklabels(labels)
    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set( color='#7570b3', linewidth=2)
        # change fill color
        box.set( facecolor = '#1b9e77' )
    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)
    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    ## change the style of fliers and their fill
    # for flier in bp['fliers']:
    #     flier.set(marker='o', color='#e7298a', alpha=1)



    plt.ylabel("time (s)")
    plt.xlabel("puzzle size")
    plt.title("Solve times per puzzle size")
    plt.show()


if __name__ == "__main__":
    PRINT_SIZES = False

    plt.style.use('seaborn-dark')

    total_solved = 0
    total_unsolved = 0

    # get all filenames sorted from puzzles folder
    filenames = glob.glob("puzzles/*.txt")
    filenames.sort()

    solve_times = []

    sizes = Counter()
    cdict_unsolved = Counter()
    cdict_solved = Counter()
    ldict_times = defaultdict(lambda : [])

    run_start = time.time()

    for i in range(len(filenames)):
        # CREATE NEW SOLVER
        solver = Solver(filenames[i])

        sizes[solver.size] += 1
        if PRINT_SIZES:
            continue

        # PRINT STATUS
        try:
            perc = (total_solved / (total_solved + total_unsolved))*100
        except:
            perc = 100
        print("running test {} / {}   ({} / {} found so far... {}%)".format(str(i+1), str(len(filenames)), str(total_solved), str(total_solved + total_unsolved), perc))

        # PUZZLE SIZE RANGE TO SOLVE
        if  not (81 <= solver.size <= 81):
            print("skipping size {}...".format(solver.size))
            continue

        solver.print_puzzle()
        # solver.solve(60)
        cProfile.run('solver.solve(30)')
        solver.print_state()

        # SOLVED OR UNSOLVED?
        if solver.verify_solution():
            print("SOLVED")
            total_solved += 1
            solve_times.append(solver.solve_time)
            cdict_solved[solver.size] += 1
            ldict_times[solver.size].append(solver.solve_time)
        else:
            print("unsolved...")
            total_unsolved += 1
            cdict_unsolved[solver.size] += 1

        print(solver.solve_time)
        print("")

    if PRINT_SIZES:
        print(sizes.most_common())
        labels = []
        values = []
        for (k, v) in sizes.most_common():
            labels.append(k)
            values.append(v)
        cmap = plt.get_cmap('Spectral')
        colors = [cmap(i) for i in np.linspace(0, 1, len(labels))]
        patches, texts = plt.pie(values, colors=colors, shadow=True, startangle=90)
        plt.legend(patches, labels, loc="best")
        plt.axis('equal')
        plt.title('Puzzle size breakdown (out of 621)')
        plt.tight_layout()
        plt.show()

    else:
        print("total solved: %d" % total_solved)
        print("total unsolved: %d" % total_unsolved)
        try:
            print("avg solve time: {:.4f}".format(sum(solve_times)/len(solve_times)))
        except:
            print("avg solve time: None")

        # make graphs

        # GRAPH 1 - percentage of each size solved under 1 minute
        labels = []
        data = []
        for k, v in sorted(sizes.items(), key=lambda item: (item[0], item[1])):
            try:
                data.append(cdict_solved[k]/(cdict_solved[k]+cdict_unsolved[k]))
                labels.append(k)
            except:
                continue

        do_bargraph(data, labels)

        # GRAPH 2 - boxplot of solve times per size
        labels = []
        data = []
        for k, v in sorted(sizes.items(), key=lambda item: (item[0], item[1])):
            if len(ldict_times[k]) != 0:
                labels.append(k)
                data.append(ldict_times[k])
        print(data)
        print(labels)

        do_boxplot(data, labels)


    print("run time: {:.2f}".format(time.time() - run_start))
