# multiAgents.py
# --------------
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


import random

import util
from game import Agent
from game import Directions


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        scaleImportantDistanceGhost = [0] * 1000
        scaleImportantDistanceGhost = [0] * 10000
        scaleImportantDistanceGhost[0] = -99999999
        scaleImportantDistanceGhost[1] = -80000000
        scaleImportantDistanceGhost[2] = -70000000
        scaleImportantDistanceGhost[3] = -60000000
        distance = bfs(successorGameState, newPos)

        if newPos in currentGameState.getCapsules():
            return 99999999

        totalDistanceFromGhost = 0
        idxGhost = -1
        for ghost in newGhostStates:
            xG, yG = int(ghost.getPosition()[0]), int(ghost.getPosition()[1])
            idxGhost += 1
            isScared = 10
            if newScaredTimes[idxGhost] > 2:
                isScared = -1
            totalDistanceFromGhost = totalDistanceFromGhost + scaleImportantDistanceGhost[distance[(xG, yG)]] * isScared

        minDistanceFromFood = 99999999
        for food in currentGameState.getFood().asList():
            minDistanceFromFood = min(minDistanceFromFood, distance[food])

        score = -minDistanceFromFood * 1000 + totalDistanceFromGhost
        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='betterEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getScore(self, gameState, agentIndex, depth):
        # Base case
        if depth == self.depth:
            return self.evaluationFunction(gameState), Directions.STOP
        if gameState.isWin():
            return self.evaluationFunction(gameState), Directions.STOP
        if gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP

        # Initialization
        legal_actions = gameState.getLegalActions(agentIndex=agentIndex)
        result_action = Directions.STOP
        if agentIndex == 0:  # Max
            result_score = -9999999
        else:
            result_score = 9999999
        successor_agent_index = agentIndex + 1
        successor_depth = depth
        if successor_agent_index == gameState.getNumAgents():
            successor_agent_index = 0
        if agentIndex == gameState.getNumAgents() - 1:
            successor_depth += 1

        # Recursion
        for action in legal_actions:
            successor_game_state = gameState.generateSuccessor(agentIndex, action)
            evaluation_score = self.getScore(successor_game_state, successor_agent_index,
                                             successor_depth)
            if agentIndex == 0:  # Max
                # if successor_game_state.isWin():
                #     return 0, action
                if result_score < evaluation_score[0]:
                    result_score = evaluation_score[0]
                    result_action = action
            else:
                if result_score > evaluation_score[0]:
                    result_score = evaluation_score[0]
                    result_action = action
        return result_score, result_action

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        _, result_action = self.getScore(gameState, 0, 0)
        return result_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getScore(self, gameState, agentIndex, alpha, beta, depth):
        # Base case
        if depth == self.depth:
            return self.evaluationFunction(gameState), Directions.STOP
        if gameState.isWin():
            return self.evaluationFunction(gameState), Directions.STOP
        if gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP

        # Initialization
        successor_agent_index = agentIndex + 1
        successor_depth = depth
        if successor_agent_index == gameState.getNumAgents():
            successor_agent_index = 0
        if agentIndex == gameState.getNumAgents() - 1:
            successor_depth += 1
        legal_actions = gameState.getLegalActions(agentIndex=agentIndex)
        result_action = Directions.STOP

        # Max
        if agentIndex == 0:
            result_score = -99999999
            for action in legal_actions:
                successor_game_state = gameState.generateSuccessor(agentIndex, action)
                # if successor_game_state.isWin():
                #     return 0, action
                # if successor_game_state.getPacmanPosition() in gameState.getCapsules():
                #     return 0, action
                evaluation_score = self.getScore(successor_game_state, successor_agent_index, alpha, beta,
                                                 successor_depth)
                if result_score < evaluation_score[0]:
                    result_score = evaluation_score[0]
                    result_action = action
                if result_score > beta:
                    return result_score, result_action
                alpha = max(alpha, result_score)
            return result_score, result_action

        # Min
        result_score = 99999999
        for action in legal_actions:
            successor_game_state = gameState.generateSuccessor(agentIndex, action)
            evaluation_score = self.getScore(successor_game_state, successor_agent_index, alpha, beta,
                                             successor_depth)
            if result_score > evaluation_score[0]:
                result_score = evaluation_score[0]
                result_action = action
            if result_score < alpha:
                return result_score, result_action
            beta = min(beta, result_score)
        return result_score, result_action

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        _, result_action = self.getScore(gameState, 0, -99999999, 99999999, 0)
        return result_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getScore(self, gameState, agentIndex, depth):
        # Base case
        if depth == self.depth:
            return self.evaluationFunction(gameState), Directions.STOP
        if gameState.isWin():
            return self.evaluationFunction(gameState), Directions.STOP
        if gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP

        # Initialization
        legal_actions = gameState.getLegalActions(agentIndex=agentIndex)
        result_action = Directions.STOP
        if agentIndex == 0:  # Max
            result_score = -9999999
        else:  # Min
            result_score = 0
        successor_agent_index = agentIndex + 1
        successor_depth = depth
        if successor_agent_index == gameState.getNumAgents():
            successor_agent_index = 0
        if agentIndex == gameState.getNumAgents() - 1:
            successor_depth += 1

        # Recursion
        for action in legal_actions:
            successor_game_state = gameState.generateSuccessor(agentIndex, action)
            evaluation_score = self.getScore(successor_game_state, successor_agent_index, successor_depth)
            if agentIndex == 0:  # Max
                # if successor_game_state.isWin():
                #     return 0, action
                if result_score < evaluation_score[0]:
                    result_score = evaluation_score[0]
                    result_action = action
            else:
                result_score = result_score + (1 / len(legal_actions)) * evaluation_score[0]
                result_action = action
        return result_score, result_action

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        _, result_action = self.getScore(gameState, 0, 0)
        return result_action


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    pacman_position = currentGameState.getPacmanPosition()
    current_food = currentGameState.getFood()
    current_capsules = currentGameState.getCapsules()
    ghost_states = currentGameState.getGhostStates()
    scared_time = [ghostState.scaredTimer for ghostState in ghost_states]
    scaleImportantDistanceGhost = [0] * 10000
    scaleImportantDistanceGhost[0] = -99999999
    scaleImportantDistanceGhost[1] = -80000000
    scaleImportantDistanceGhost[2] = -70000000
    scaleImportantDistanceGhost[3] = -60000000
    numberFoodRemain = current_food.count()
    if numberFoodRemain == 0:
        return 9999999
    if pacman_position in current_capsules:
        return 99999999
    distance = bfs(currentGameState, pacman_position)
    minDistanceFromFood = 99999999
    for food in currentGameState.getFood().asList():
        minDistanceFromFood = min(minDistanceFromFood, distance[food])
    distanceToGhost = 0
    idxGhost = -1
    for ghost in ghost_states:
        xG, yG = int(ghost.getPosition()[0]), int(ghost.getPosition()[1])
        idxGhost += 1
        isScared = 10
        if scared_time[idxGhost] > 1:
            isScared = -1
        distanceToGhost = distanceToGhost + scaleImportantDistanceGhost[distance[(xG, yG)]] * isScared
    score = -numberFoodRemain * 1000 - minDistanceFromFood * 100 + distanceToGhost
    return score


def bfs(gameState, pacmanPosition):
    from util import Queue, Counter
    from game import Actions
    walls = gameState.getWalls()
    w = gameState.data.layout.width
    h = gameState.data.layout.height

    distance = Counter()
    qu = Queue()
    qu.push(pacmanPosition)
    distance[pacmanPosition] = 0

    while not qu.isEmpty():
        position = qu.pop()
        for action in [Directions.NORTH, Directions.EAST, Directions.WEST, Directions.SOUTH]:
            x, y = position
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if nextx < 0 or nexty < 0 or nextx >= w or nexty >= h:
                continue
            if not walls[nextx][nexty]:
                nextPosition = (nextx, nexty)
            else:
                continue
            if distance[nextPosition] == 0 and nextPosition != pacmanPosition:
                distance[nextPosition] = 99999999
            if distance[nextPosition] > distance[position] + 1:
                distance[nextPosition] = distance[position] + 1
                qu.push(nextPosition)
    return distance


# Abbreviation
better = betterEvaluationFunction
