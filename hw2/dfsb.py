import sys
import math
import copy
import time


class CSP:

    def __init__(self, mode, num_variables, num_domains, num_contraints,
                 constraints):
        self.mode = mode
        self.num_variables = int(num_variables)
        self.constraints = constraints
        self.num_domains = int(num_domains)
        self.variables = []
        for i in range(0, int(self.num_variables)):
            self.variables.append(i)
        #print(self.variables)
        self.domains = {
            key: [i for i in range(0, int(num_domains))]
            for key in range(0, int(num_variables))
        }
        self.adjacents = {key: [] for key in range(0, int(self.num_variables))}
        for c in self.constraints:
            self.adjacents[int(c[0])].append(int(c[1]))
            self.adjacents[int(c[1])].append(int(c[0]))
        #print(self.adjacents)

        domain_v = [i for i in range(int(num_domains))]

        self.domains_current = {
            key: [i for i in range(0, int(num_domains))]
            for key in range(0, int(num_variables))
        }

    # print("Variables = " + str(self.variables))
    # print("Neighbours =  " + str(self.adjacents))
    # print("Domains = "+ str(self.domains))


def findMostRestrictedVariable(vars_unassigned, csp):
    min = float('inf')
    max_neighbours = float('-inf')
    max_neighbours_var = float('-inf')

    #Most restricted variable will be the one with minimum current domain values and
    # one with the max neighbours

    #Find the length of  minimum domain
    for var in vars_unassigned:
        if min > len(csp.domains_current[var]):
            min = len(csp.domains_current[var])

    #Gather all the variables with min current domain in a list
    mrv_list = []
    for var in vars_unassigned:
        if len(csp.domains_current[var]) == min:
            mrv_list.append(var)

    #From these variables in mrv_list, pick the one that has the most neighbours
    for var in mrv_list:
        if len(csp.adjacents[var]) > max_neighbours:
            max_neighbours = len(csp.adjacents[var])
            max_neighbours_var = var

    return max_neighbours_var


def selectUnassignedVariable(assignment, csp):
    #Return the next unassigned variable
    if csp.mode == 0:
        for var in csp.variables:
            if var not in assignment:
                return var
    #return the most restricted unassigned variable
    if csp.mode == 1:
        vars_unassigned = []
        for var in csp.variables:
            if var not in assignment:
                vars_unassigned.append(var)
        mrv = findMostRestrictedVariable(vars_unassigned, csp)
        return mrv


def orderDomainValues(var, assignment, csp):

    dict_lcv = {}
    for key in csp.domains_current[var]:
        dict_lcv.update({key: 0})
    '''
        Ordering the domain values in the increasing order of the neighbours that get affected
        This is done by maintaining the count of neighbours having this value in their current domain
        this information is kept in the dictionary value:neighbours_count
        Domain values are returned in the increasing order of the neighbour_count.
    '''
    for neighbour in csp.adjacents[var]:
        for value in csp.domains_current[var]:
            if value in csp.domains_current[neighbour]:
                dict_lcv[value] += 1

    lc_value = sorted(dict_lcv, key=dict_lcv.get)
    return [v for v in lc_value]


#Check the neighbours assignments, they should not be inconsistent with this variables's assignment.
def isConsistent(value, var, assignment, csp):
    for neighbour in csp.adjacents[var]:
        if (neighbour in assignment):
            if value == assignment.get(neighbour, None) and (assignment.get(
                    neighbour, None) == value):
                return False
    return True


def AC3(csp, Q):
    while len(Q) > 0:
        #pop an arc from the queue
        (head, tail) = Q.pop(0)
        #if no consistent value remains in head for a value picked for tail, remove the inconsitent value from head
        if len(csp.domains_current[tail]) == 1:
            if csp.domains_current[tail][0] in csp.domains_current[head]:
                csp.domains_current[head].remove(csp.domains_current[tail][0])
                if len(csp.domains_current[head]) == 0:
                    return False
                for next in csp.adjacents[head]:
                    if next != tail:
                        Q.append((next, head))
                    else:
                        continue
    return True


steps = 0


def searchRecursive(assignment, csp):
    global steps
    steps += 1
    #Check if the assignment to all the variables is done, if yes, return the assignment
    if len(assignment) == csp.num_variables:
        return assignment
    #Choose an unassigned variable to be next assigned
    variable = selectUnassignedVariable(assignment, csp)
    #For each value in th domain, check if it is consistent with the neighbours, if yes, pick the value
    for value in range(0, csp.num_domains):
        if isConsistent(value, variable, assignment, csp):
            assignment[variable] = value
            #Now build the further assignments on this consistent assignment
            res = searchRecursive(assignment, csp)
            if res != False:
                return res
            #if failure, remove the current variable assignment
            del assignment[variable]
    return False


def searchImproved(assignment, csp):
    global steps
    steps += 1
    if len(assignment) == csp.num_variables:
        return assignment
    domain_values_copy = copy.deepcopy(csp.domains_current)
    #Find the most restricted Variable (MRV)
    variable = selectUnassignedVariable(assignment, csp)
    # Get the values in the least constraining order
    domain_values = copy.deepcopy(orderDomainValues(variable, assignment, csp))

    for value in domain_values:
        if isConsistent(value, variable, assignment, csp):
            csp.domains_current[variable] = [value]
            assignment[variable] = value
            # Once the assignment is done, put all the arcs in the queue to
            # check if they are still consistent
            q = []
            for head in csp.adjacents[variable]:
                q.append((head, variable))
            #Do the consistency check
            arc_consistent = AC3(csp, q)
            steps += 1
            if arc_consistent != False:
                res = searchImproved(assignment, csp)
                if res != False:
                    return res
        #retrieve the domain values from the backup
        csp.domains_current = copy.deepcopy(domain_values_copy)
        #remove this current variable assignment
        del assignment[variable]
    return False


if __name__ == '__main__':
    input_file = ""
    output_file = ""
    mode = int(sys.argv[3])
    if len(sys.argv) == 4:
        algorithm = "Dfsb" if mode == 0 else "ImprovedDfsb"
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        print("Please enter all valid arguments")
        print("python3 dfsb.py <input_file> <output_file> <mode_flag>")
        sys.exit(1)

    print('Algorithm = ' + algorithm)

    fin = open(input_file, 'r')
    line = fin.readline().strip()
    top_row = line.split('\t')
    num_variables, num_constraints, num_domains = top_row[0], top_row[
        1], top_row[2]

    constraints = []
    for i in range(int(num_constraints)):
        line = fin.readline().strip()
        constraints.append(line.split('\t'))

    #Create a CSP object that describes the problem
    csp = CSP(mode, num_variables, num_domains, num_constraints, constraints)
    res = False
    final_result = {}

    #time tracking
    start_time = time.time() * 1000
    if algorithm == "Dfsb":
        res = searchRecursive(final_result, csp)
    elif algorithm == "ImprovedDfsb":
        res = searchImproved(final_result, csp)
    end_time = time.time() * 1000
    print("Time taken(ms) = " + str(end_time - start_time))

    fout = open(output_file, 'w')
    if res == False:
        print("No answer")
        fout.write("No answer")
    else:
        print("\n\n Possible Valid assignment : \n")
        print(final_result)
        print("\n Steps = " + str(steps))

    #Write the output to the file

    for color in final_result:
        fout.write(str(final_result[color]))
        fout.write('\n')
