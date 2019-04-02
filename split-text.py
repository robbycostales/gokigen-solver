from __future__ import division
from __future__ import print_function

# Author: Robby Costales
# Date: 2018-04-15
# Language: Python 2

import glob

# Purpose: splitting Gokigen.txt into separate files for easier use and reading

def splitPuzzles(fileName):
    f = open(fileName, "r")
    contents = f.read()
    f.close()
    singulars = contents.split("end")

    # write all singulars to files in /puzzles/ dir
    # -1 because last one is blank
    for i in range(len(singulars)-1):
        newFile = open("puzzles/{}.txt".format(str(i+1).zfill(3)), "w+")
        newFile.write(singulars[i])
        newFile.write("moves")
        newFile.close()

if __name__ == "__main__":
    splitPuzzles("og-text/Gokigen.txt")
