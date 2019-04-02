



def read_puzzle(filename):
    """
    Function for getting problem and solution information from each puzzle in the /puzzles/ directory

    Args:
        filename : name of file
    Returns:
        problem, solution -- in form of nested lists
    """
    f = open(filename, "r")
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
