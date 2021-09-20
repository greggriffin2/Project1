# search.py
# ---------
# Greg Griffin and David Novak 
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
    "*** Sample solution of DFS is provided ***"

    from util import Stack

    stackXY = Stack()

    visited = []  # Visited states
    path_to_xy = []  # Every state keeps it's path from the starting state

    # Check if initial state is goal state #
    initial_state = problem.getStartState()
    if problem.isGoalState(initial_state):
        return []

    # Start from the beginning and find a solution, path is an empty list #
    stackXY.push((initial_state, []))

    while True:

        # Terminate condition: can't find solution #
        if stackXY.isEmpty():
            return []

        # Get information of current state #
        xy, path_to_xy = stackXY.pop()  # Take position and path
        visited.append(xy)

        # Terminate condition: reach goal #
        if problem.isGoalState(xy):
            return path_to_xy

        # Get successors of current state #
        succ = problem.getSuccessors(xy)

        # Add new states in stack and fix their path #
        if succ:
            for item in succ:
                if item[0] not in visited:  # If this is a new position we have not visited...
                    newPath = path_to_xy + [item[1]]  # Calculate new path
                    stackXY.push((item[0], newPath))


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    explored = set()  # set of explored nodes
    path_to_curr = []  # a list of actions required to get to the current node
    q = util.Queue()  # Queue used to store nodes to be expanded

    # Check if initial state is goal state #
    initial_state = problem.getStartState()
    if problem.isGoalState(initial_state):
        return []

    # Place the start state into the queue, along with the list of actions used to get to this node
    q.push((initial_state, []))

    # while the queue is not empty
    while q:

        # Pop a node from the queue, assigning variables to its xy coordinates and list of actions
        nodeState, path_to_curr = q.pop()

        # Add the xy coordinate to the set of explored nodes
        explored.add(nodeState)

        # Get the successors to the current nodeState
        succ = problem.getSuccessors(nodeState)

        # For every successor of the current node that has not been already explored...
        if succ:
            for child in succ:
                if child[0] not in explored:

                    # Goal Test to terminate search and return the action steps to reach the goal
                    if problem.isGoalState(child[0]):
                        path_to_curr.append(child[1])
                        return path_to_curr

                    # Otherwise, calculate the new path to the current child and push it and its xy coordinates into
                    # the queue
                    newPath = path_to_curr + [child[1]]
                    q.push((child[0], newPath))

    # Only called if the queue is empty and a goal state has not been found, returns an empty list of actions
    return []


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    explored = set()  # stores previously explored nodes
    path_to_curr = []  # a list of actions required to get to the current node
    q = util.PriorityQueue()  # the priority queue used to store nodes to be expanded

    # Check if initial state is goal state #
    initial_state = problem.getStartState()
    if problem.isGoalState(initial_state):
        return []

    # Push into the queue a triple of the start state's xy coords, a list of the actions required to get to it,
    # and its cost. This has a priority of 0
    q.push((initial_state, [], 0), 0)

    # while the queue is not empty
    while q:

        # Pop a node from the queue, assigning variables to its xy coordinates, list of actions, and cost
        nodeState, path_to_curr, cost = q.pop()

        # Add the xy coordinate to the set of explored nodes
        explored.add(nodeState)

        # Get the successors to the current node
        succ = problem.getSuccessors(nodeState)

        # For every child of the current node that has not been already explored...
        if succ:
            for child in succ:
                if child[0] not in explored:

                    # Goal Test to terminate search and return the action steps needed to reach the goal
                    if problem.isGoalState(child[0]):
                        path_to_curr.append(child[1])
                        return path_to_curr

                    # Add its cost to its parent's to get its new cost (priority)
                    newCost = cost + child[2]

                    # Calculate the new path to the current child
                    newPath = path_to_curr + [child[1]]

                    # If child already in queue with higher cost, update its cost (priority) and path
                    # If child already in queue with equal or lower cost, do nothing.
                    # If child not in queue, push it into the queue
                    q.update((child[0], newPath, newCost), newCost)

    # Only called if the queue is empty and a goal state has not been found, returns an empty list of actions
    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    explored = set()  # stores previously explored nodes
    path_to_curr = []  # a list of actions required to get to the current node
    q = util.PriorityQueue()  # the priority queue used to store nodes to be expanded

    # Check if initial state is goal state
    initial_state = problem.getStartState()
    if problem.isGoalState(initial_state):
        return []

    # Push into the queue a triple of the initial state's xy coords, a list of the actions required to get to it,
    # and its cost.
    q.push((initial_state, [], 0), 0)

    # while the queue is not empty
    while q:

        # Pop a node from the queue, assigning variables to its xy coordinates, list of actions, and cost
        nodeState, path_to_curr, cost = q.pop()

        # Add the xy coordinate to the set of explored nodes
        explored.add(nodeState)

        # Get the successors to the current node
        succ = problem.getSuccessors(nodeState)

        # For every child of the current node that has not been already explored...
        if succ:
            for child in succ:
                if child[0] not in explored:

                    # Goal Test to terminate search and return the action steps needed to reach the goal
                    if problem.isGoalState(child[0]):
                        path_to_curr.append(child[1])
                        return path_to_curr

                    # Calculate the cost (priority) of this child by the cost function:
                    # F(n) = g(n) + h(n)
                    newCost = cost + child[2] + heuristic(child[0], problem)

                    # Calculate the new path to the current child
                    newPath = path_to_curr + [child[1]]

                    # If child already in queue with higher cost, update its cost and path
                    # If child already in queue with equal or lower cost, do nothing.
                    # If child not in queue, push it into the queue
                    q.update((child[0], newPath, newCost), newCost)

    # Only called if the queue is empty and a goal state has not been found, returns an empty list of actions
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
