# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def tracePath(start_state, goal_state, trace):
    direction = list()
    while goal_state != start_state:
        direction.append(trace[goal_state][1])
        goal_state = trace[goal_state][0]
    direction.reverse()
    return direction


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from util import Counter, Stack
    trace = Counter()
    stack = Stack()
    visitedState = list()
    stack.push(problem.getStartState())
    goal_state = None
    while not stack.isEmpty():
        state = stack.pop()
        if state in visitedState:
            continue
        visitedState.append(state)
        if problem.isGoalState(state):
            goal_state = state
            break
        for successor in problem.getSuccessors(state):
            nextState, action, cost = successor
            if nextState not in visitedState:
                trace[nextState] = (state, action)
            stack.push(nextState)
    return tracePath(problem.getStartState(), goal_state, trace)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue, Counter
    queue = Queue()
    trace = Counter()
    queue.push(problem.getStartState())
    trace[problem.getStartState()] = -1
    goal_state = None
    while not queue.isEmpty():
        state = queue.pop()
        if problem.isGoalState(state):
            goal_state = state
            break
        for successor in problem.getSuccessors(state):
            next_state, action, cost = successor
            if trace[next_state] == 0:
                trace[next_state] = (state, action)
                queue.push(next_state)
    return tracePath(problem.getStartState(), goal_state, trace)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue, Counter
    trace = Counter()
    pq = PriorityQueue()
    visited_state = list()
    pq.push(problem.getStartState(), 0)
    goal_state = None
    cost_state = Counter()
    while not pq.isEmpty():
        state = pq.pop()
        if state in visited_state:
            continue
        visited_state.append(state)
        if problem.isGoalState(state):
            goal_state = state
            break
        for successor in problem.getSuccessors(state):
            nextState, action, cost = successor
            if nextState in visited_state:
                continue
            if cost_state[nextState] == 0 or cost_state[nextState] > cost_state[state] + cost:
                trace[nextState] = (state, action)
                cost_state[nextState] = cost_state[state] + cost
                pq.update(nextState, cost_state[nextState])
    return tracePath(problem.getStartState(), goal_state, trace)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue, Counter
    pq = PriorityQueue()
    f_state = Counter()
    g_state = Counter()
    trace = Counter()
    visited_state = list()
    goal_state = None
    f_state[problem.getStartState()] = heuristic(problem.getStartState(), problem)
    pq.push(problem.getStartState(), f_state[problem.getStartState()])
    while not pq.isEmpty():
        state = pq.pop()
        if state in visited_state:
            continue
        if problem.isGoalState(state):
            goal_state = state
            break
        visited_state.append(state)
        for successor in problem.getSuccessors(state):
            next_state, action, cost = successor
            if next_state in visited_state:
                continue
            if f_state[next_state] == 0 or f_state[next_state] > g_state[state] + cost + heuristic(next_state, problem):
                g_state[next_state] = g_state[state] + cost
                f_state[next_state] = g_state[next_state] + heuristic(next_state, problem)
                trace[next_state] = (state, action)
                pq.update(next_state, f_state[next_state])
    return tracePath(problem.getStartState(), goal_state, trace)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
