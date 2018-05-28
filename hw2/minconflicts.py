import sys
import math
import copy
import random
import time


class CSP:

    def __init__(self, num_variables, num_domains, num_contraints, constraints):

        self.attempts = 0
        self.limit = 20

        self.num_variables = int(num_variables)
        self.constraints = constraints
        self.domain = [v for v in range(int(num_domains))]

        self.variables = []
        for i in range(0, int(self.num_variables)):
            self.variables.append(i)

        self.adjacents = {key: [] for key in range(0, int(self.num_variables))}
        for c in self.constraints:
            self.adjacents[int(c[0])].append(int(c[1]))
            self.adjacents[int(c[1])].append(int(c[0]))
        for c in self.constraints:
            self.adjacents[int(c[0])].append(int(c[1]))
            self.adjacents[int(c[1])].append(int(c[0]))


# Randomly chose a variable from the set of conflicted variables
def getRandomConflictingVariable(assignment, csp):
    #if the limit crossed for this random assignment, return any variable at random
    if csp.attempts > csp.limit:
        return random.randrange(0, csp.num_variables)

    #list of the conflicting variables
    conflicting_variables = []
    #check conflicting assignments and append the corresponding variables to the list
    for v in range(0, csp.num_variables):
        for neighbour in csp.adjacents[v]:
            if assignment[neighbour] == assignment[v]:
                conflicting_variables.append(v)
                break
    #radomly return one of the conflicting variables
    return random.choice(conflicting_variables)


#Finds the value val for var that minimizes conflicts
def getLeastConflictingValue(var, assignment, csp):

    if csp.attempts > csp.limit:
        csp.attempts = 0
        domain_siz = len(csp.domain)
        least_conflict_value = csp.domain[random.randrange(0, domain_siz)]
        return least_conflict_value

    conflicts = []
    least_conflict_count = float('inf')
    for val in csp.domain:
        conf_count = 0
        #try this new assignment by assigning var = val
        new_assignment = copy.deepcopy(assignment)
        new_assignment[var] = val

        #Count the number of conflicts if we assign val to this variable.
        for arc in csp.constraints:
            if new_assignment[int(arc[0])] == new_assignment[int(arc[1])]:
                conf_count = conf_count + 1
        #append this count to the conflicts list
        conflicts.append(conf_count)

        #Keep a check on the minimum conflict count
        if conf_count < least_conflict_count:
            least_conflict_count = conf_count

    #Gather all the values that gave the least_conflict_count and randomly return one of those
    possible_values = [
        i for i, count in enumerate(conflicts) if count == least_conflict_count
    ]
    return random.choice(possible_values)


steps = 0


def minConflicts(assignment, csp):
    global steps

    for var in csp.variables:
        val = random.randrange(0, len(csp.domain))
        assignment[var] = val

    random_restart = 0
    while random_restart < 100:
        steps += 1

        ##check if this random assignment is the actual solution, if yes, return the assignment
        res = True
        for arc in csp.constraints:
            if assignment[int(arc[0])] == assignment[int(arc[1])]:
                res = False

        if res == True:
            return assignment

        # if we have crossed the maximum limit of trying different values for this assignment, do a random restart
        if (csp.attempts > csp.limit):
            for var in csp.variables:
                val = random.randrange(0, len(csp.domain))
                assignment[var] = val
                random_restart += 1
            csp.attempts = 0

        csp.attempts += 1
        #We did not find a solution yet, pick a random variable from amongst the conflicting variables
        #This will be the most resctricted variable
        variable = getRandomConflictingVariable(assignment, csp)
        # Now pick a value to assign to this variable
        # This will be the least conflicting value.
        value = getLeastConflictingValue(variable, assignment, csp)
        assignment[variable] = value
    return False


if __name__ == '__main__':
    input_file = ""
    output_file = ""
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        print("Please enter all valid arguments")
        print("python3 dfsb.py <input_file> <output_file>")
        sys.exit(1)

    print('Input file = ' + input_file)
    print('Output_file = ' + output_file)

    fin = open(input_file, 'r')
    line = fin.readline().strip()
    top_row = line.split('\t')
    num_variables, num_constraints, num_domains = top_row[0], top_row[
        1], top_row[2]
    #print("Domains " + num_domains)
    constraints = []
    for i in range(int(num_constraints)):
        line = fin.readline().strip()
        constraints.append(line.split('\t'))

    #Create a CSP object that describes the problem
    csp = CSP(num_variables, num_domains, num_constraints, constraints)

    final_res = {}
    #Start the timer
    start_time = time.time() * 1000
    res = minConflicts(final_res, csp)
    #Record the current time
    end_time = time.time() * 1000
    print("Time taken(ms) = " + str(end_time - start_time))

    fout = open(output_file, 'w')
    if res == False:
        print("No answer")
        fout.write("No answer")
    else:
        print("\n Possible valid assignment is: \n")
        print(final_res)
        print("\nSteps = " + str(steps))

    #Print the output in the file

    for color in final_res:
        fout.write(str(final_res[color]))
        fout.write('\n')
