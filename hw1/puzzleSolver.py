from queue import PriorityQueue
import sys
import math
#import time

true = 1
false = 0


def manhattan_heu(state, p_size):
    heu = 0
    for i, r in enumerate(state):
        for j, val in enumerate(r):
            if val == None:
                continue
            heu = heu + abs(((val - 1) / p_size) - i) + abs((
                (val - 1) % p_size) - j)
    return heu


def misplaced_tiles_heu(state, p_size):
    heu = 0
    tile_number = 1
    for i, row in enumerate(state):
        for j, tile in enumerate(row):
            if i == p_size - 1 and j == p_size - 1:
                goal_tile = None
            else:
                goal_tile = tile_number
            tile_number += 1
            if tile != goal_tile:
                heu += 1
    return heu


class AStar:

    def __init__(self, start_state, heu, size):
        self.explored_nodes = 0
        self.result = None
        self.p_size = size
        self.heu = heu
        self.goal_state = []
        self.goal_state = [
            list(range(i, i + int(size)))
            for i in range(1,
                           int(size)**2, int(size))
        ]
        self.goal_state[size - 1][size - 1] = None
        self.goal_state = tuple(tuple(row) for row in self.goal_state)
        self.start_node = State(
            self, start_state, parent_node=None, action=None)
        self.explored = {}
        self.frontier = PriorityQueue()

    def checkIfGoal(self, state):
        if self.goal_state == state:
            return true
        else:
            return false

    def getSuccessor(self, state, action):

        row, col = self.getNoneIJ(state)
        successor = list(list(r) for r in state)

        if action == 'U':
            tmp = successor[row - 1][col]
            successor[row - 1][col] = successor[row][col]
            successor[row][col] = tmp
        elif action == 'D':
            tmp = successor[row + 1][col]
            successor[row + 1][col] = successor[row][col]
            successor[row][col] = tmp
        elif action == 'L':
            tmp = successor[row][col - 1]
            successor[row][col - 1] = successor[row][col]
            successor[row][col] = tmp
        elif action == 'R':
            tmp = successor[row][col + 1]
            successor[row][col + 1] = successor[row][col]
            successor[row][col] = tmp

        return tuple(tuple(r) for r in successor)

    def expand(self, curr_node):
        for action in curr_node.getLegitActions(self, curr_node):
            next_state = self.getSuccessor(curr_node.curr_state, action)
            new_node = State(self, next_state, curr_node, action)
            self.count += -1
            self.frontier.put((new_node.f, self.count, new_node))

    def solvePuzzle(self):
        result = []
        self.count = 0
        self.count += -1

        self.frontier.put((self.start_node.f, self.count, self.start_node))
        while (self.frontier.qsize() != 0):
            p, _, curr_node = self.frontier.get()
            curr_state = curr_node.curr_state
            if self.checkIfGoal(curr_state):
                break
            if curr_state not in self.explored:
                self.explored[curr_state] = curr_node
                self.expand(curr_node)

        while curr_node.action:
            result.append(curr_node.action)
            curr_node = self.explored[curr_node.parent_node.curr_state]
        self.result = result[::-1]
        return self.result

    def getNoneIJ(self, state):
        none_i = None
        none_j = None
        for i, row in enumerate(state):
            for j, tile in enumerate(row):
                if (tile == None):
                    none_i = i
                    none_j = j
                    break
        return (none_i, none_j)


class State:

    def __init__(self, AStar_obj, state, parent_node, action):
        self.curr_state = state
        self.parent_node = parent_node
        self.action = action
        self.h = AStar_obj.heu(self.curr_state, AStar_obj.p_size)
        self.g = 0
        if parent_node is None:
            self.g = 0
        else:
            self.g = self.parent_node.g + 1
        self.f = self.g + self.h

    def getLegitActions(self, AStar_obj, state_node):
        i, j = AStar_obj.getNoneIJ(state_node.curr_state)
        state_node.possible_actions = ['L', 'R', 'U', 'D']
        reverse_actions = {'L': 'R', 'R': 'L', 'U': 'D', 'D': 'U'}
        #topRow
        if i == 0:
            state_node.possible_actions.remove('U')
        #bottomRow
        if i == AStar_obj.p_size - 1:
            state_node.possible_actions.remove('D')
        #LeftColumn
        if j == 0:
            state_node.possible_actions.remove('L')
        #RightColumn
        if j == AStar_obj.p_size - 1:
            state_node.possible_actions.remove('R')

        if (state_node.action is not None):
            state_node.possible_actions.remove(
                reverse_actions.get(state_node.action))

        return state_node.possible_actions


class IDAStar(AStar):

    def solvePuzzle(self):
        fcutoff = self.start_node.f

        while (self.result is None):
            new_fcutoff, self.result = self.memoryBoundedSearch(
                self.start_node, fcutoff, 0, None)
            fcutoff = new_fcutoff
        return self.result

    def memoryBoundedSearch(self, state_node, fcutoff, cost, action):
        self.explored_nodes += 1
        f_value = cost + self.heu(state_node.curr_state, self.p_size)
        possible_f_value = float('inf')
        if (f_value > fcutoff):
            return f_value, None

        if self.checkIfGoal(state_node.curr_state):
            return f_value, []

        possible_actions = state_node.getLegitActions(self, state_node)
        for action in possible_actions:
            next_state = self.getSuccessor(state_node.curr_state, action)
            new_node = State(self, next_state, state_node, action)
            new_fvalue, res = self.memoryBoundedSearch(new_node, fcutoff,
                                                       cost + 1, action)
            if res is None:
                possible_f_value = min(new_fvalue, possible_f_value)
            else:
                return fcutoff, [action] + res
        return possible_f_value, None


if __name__ == '__main__':
    #start_time = time.time()
    algorithm = 0
    n = 0
    h = 0
    input_file = ""
    output_file = ""

    if len(sys.argv) == 6:
        algorithm = AStar if sys.argv[1] == '1' else IDAStar
        n = int(sys.argv[2])
        h = manhattan_heu if sys.argv[3] == '1' else misplaced_tiles_heu
        input_file = sys.argv[4]
        output_file = sys.argv[5]
    else:
        print("Please enter all valid arguments")
        print(
            "python puzzleSolver.py <#Algorithm> <N> <H> <INPUT_FILE_PATH> <OUTPUT_FILE_PATH>"
        )
        sys.exit(1)

    print('Algorithm(1: A* 2: IDA*) = ' + sys.argv[1])
    print('Puzzle size = ' + str(n))
    print('Heuristic(1: Manhattan Distance 2: Number of Misplaced Tiles) = ' +
          sys.argv[3])
    print('Input file = ' + input_file)
    print('Output_file = ' + output_file)

    init_state = []
    fin = open(input_file, 'r')
    for line in fin.readlines():
        lines = line.rstrip('\n').split('\n')
        for r in lines:
            tiles = r.split(',')
        init_state.append([int(x) if x else None for x in tiles])

    init_state = tuple(tuple(r) for r in init_state)
    print("Start state" + str(init_state) + '\n')
    fin.close()
    obj = algorithm(init_state, h, n)
    obj.solvePuzzle()
    fout = open(output_file, 'w')
    fout.write(','.join(obj.result))
    print(','.join(obj.result))
    fout.write('\n')
    fout.close()
