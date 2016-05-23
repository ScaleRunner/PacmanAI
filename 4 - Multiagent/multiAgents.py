# 4455770       Dennis Verheijden       KI
# 4474139       Remco van der Heijden   KI

# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random
import util
import math

from searchAgents import ClosestDotSearchAgent
from game import Agent

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
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newFood = successorGameState.getFood()
    newFoodList = newFood.asList()
    newGhostStates = successorGameState.getGhostStates()
    newGhostDistance = [manhattanDistance(ghostState.getPosition(), newPos) for ghostState in newGhostStates]
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    wallList = currentGameState.getWalls().asList()

    # Help Functions
    def minFoodDist(foodList, newPos):
        """
            Returns the minimum food distance
        """
        distance = min([manhattanDistance(food, newPos) for food in foodList.asList()])
        return distance

    def foodCount(foodList):
        """
            Returns the amount of remaining food
        """
        sum = 0
        for food in foodList.asList():
            if food:
                sum += 1
        return sum

    if successorGameState.isLose():
        return float('-Inf')
    if successorGameState.isWin():
        return float('Inf')

    # The Heuristic is roughly the sum of: The FoodDifference and the -minFoodDistance

    foodDifference = abs(foodCount(newFood) - foodCount(oldFood))
    heuristic = 0

    # You don't have to take the ghosts in account because you could just eat them or ignore them
    if sum(newScaredTimes) > 2:
        if foodDifference == 0: # Fix for pacman getting stuck in two states
            heuristic += -minFoodDist(newFood, newPos)
        heuristic += foodDifference * 10
        heuristic += 30
    else:
        if foodDifference == 0:
            heuristic += -minFoodDist(newFood, newPos)
        heuristic += foodDifference * 10
        # Ghosts are not scared, if you get to close take the ghost distance in account so you won't get eaten
        if min(newGhostDistance) < 2:
            heuristic += min(newGhostDistance) * 10
        else:
            heuristic += 30

    return heuristic

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

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent for one opponent (assignment 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    # Helper Functions
    def maxValue(state, currentDepth):
        """
            Calculates the maximum score possible for the pacman Agent
        """
        currentDepth = currentDepth + 1
        if state.isWin() or state.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(state)
        maxScore = float('-Inf')
        for pacmanAction in state.getLegalActions(0):
            maxScore = max(maxScore, minValue(state.generateSuccessor(0, pacmanAction), currentDepth, 1))
        return maxScore

    def minValue(state, currentDepth, ghostIndex):
        """
            Calculates the minimum score possible for the ghost Agent(s)
        """
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        minScore = float('Inf')
        for ghostAction in state.getLegalActions(ghostIndex):
            if ghostIndex == gameState.getNumAgents() - 1:
                minScore = min(minScore, maxValue(state.generateSuccessor(ghostIndex, ghostAction), currentDepth))
            else:
                minScore = min(minScore, minValue(state.generateSuccessor(ghostIndex, ghostAction), currentDepth, ghostIndex + 1))
        return minScore

    # Begin MiniMax
    pacmanActions = gameState.getLegalActions(0)
    maximum = float('-Inf')
    maxAction = ''
    for pacmanAction in pacmanActions:
      currentDepth = 0
      currentMax = minValue(gameState.generateSuccessor(0, pacmanAction), currentDepth, 1)
      if currentMax > maximum:
        maximum = currentMax
        #print(maximum)
        maxAction = pacmanAction
        #print(maxAction)
        #print(maximum)
    return maxAction

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning for one ghost (assignment 3)
  """

  def getAction(self, gameState):
    """
      Returns the AlphaBeta action using self.depth and self.evaluationFunction
    """
    # Helper Functions
    def maxValue(state, currentDepth, alpha, beta):
        """
            Calculates the maximum score possible for the pacman Agent
        """
        currentDepth = currentDepth + 1
        if state.isWin() or state.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(state)
        maxScore = float('-Inf')
        for pacmanAction in state.getLegalActions(0):
            maxScore = max(maxScore, minValue(state.generateSuccessor(0, pacmanAction), currentDepth, 1, alpha, beta))
            alpha = max(alpha, maxScore)
            if beta <= alpha:
                break  # prune
        return maxScore

    def minValue(state, currentDepth, ghostIndex, alpha, beta):
        """
            Calculates the minimum score possible for the ghost Agent(s)
        """
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        minScore = float('Inf')
        for ghostAction in state.getLegalActions(ghostIndex):
            if ghostIndex == gameState.getNumAgents() - 1:
                minScore = min(minScore, maxValue(state.generateSuccessor(ghostIndex, ghostAction), currentDepth, alpha, beta))
            else:
                minScore = min(minScore, minValue(state.generateSuccessor(ghostIndex, ghostAction), currentDepth, ghostIndex + 1, alpha, beta))
            beta = min(beta, minScore)
            if beta <= alpha:
                break  # prune
        return minScore

    # Begin MiniMax
    pacmanActions = gameState.getLegalActions(0)
    pacmanActions.remove("Stop")
    maximum = float('-Inf')
    alpha = float('-Inf')
    beta = float('Inf')
    maxAction = ''
    for pacmanAction in pacmanActions:
      currentDepth = 0
      currentMax = minValue(gameState.generateSuccessor(0, pacmanAction), currentDepth, 1, alpha, beta)
      if currentMax > maximum:
        maximum = currentMax
        maxAction = pacmanAction
        #print(maxAction)
        #print(maximum)
    return maxAction

class MultiAlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning for several ghosts (Extra credit assignment B)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    # Helper Functions
    def maxValue(state, currentDepth, alpha, beta):
        """
            Calculates the maximum score possible for the pacman Agent
        """
        currentDepth = currentDepth + 1
        if state.isWin() or state.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(state)
        maxScore = float('-Inf')
        for pacmanAction in state.getLegalActions(0):
            maxScore = max(maxScore, minValue(state.generateSuccessor(0, pacmanAction), currentDepth, 1, alpha, beta))
            alpha = max(alpha, maxScore)
            if beta <= alpha:
                break  # prune
        return maxScore

    def minValue(state, currentDepth, ghostIndex, alpha, beta):
        """
            Calculates the minimum score possible for the ghost Agent(s)
        """
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        minScore = float('Inf')
        for ghostAction in state.getLegalActions(ghostIndex):
            if ghostIndex == gameState.getNumAgents() - 1:
                minScore = min(minScore, maxValue(state.generateSuccessor(ghostIndex, ghostAction), currentDepth, alpha, beta))
            else:
                minScore = min(minScore, minValue(state.generateSuccessor(ghostIndex, ghostAction), currentDepth, ghostIndex + 1, alpha, beta))
            beta = min(beta, minScore)
            if beta <= alpha:
                break  # prune
        return minScore

    # Begin AlphaBeta
    pacmanActions = gameState.getLegalActions(0)
    pacmanActions.remove("Stop")
    maximum = float('-Inf')
    alpha = float('-Inf')
    beta = float('Inf')
    maxAction = ''
    for pacmanAction in pacmanActions:
      currentDepth = 0
      currentMax = minValue(gameState.generateSuccessor(0, pacmanAction), currentDepth, 1, alpha, beta)
      if currentMax > maximum:
        maximum = currentMax
        maxAction = pacmanAction
        #print(maxAction)
        #print(maximum)
    return maxAction

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (not used in this course)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function for one ghost (extra credit assignment A).

    DESCRIPTION:
    We tried to do the following calculate the distance between pacman and the food while taking walls in
    consideration. We stumbled on some problems that we couldn't fix. So we made the minFoodDist higher if there was
    a wall directly between pacman and the food. This however brought some problems on it's own.

    Further we used almost the same heuristic as the reflexagent, with some adjustments because you're evaluating
    states here and not actions.

    We stumbled on some really weird things: If pacman was at the last pellet it calculated the possible states,
    saw they were all WINS but didn't eat the pellet we are kinda lost on why this was. We also discovered that using
    "float(Inf)" was worse than just using a really high value (why?). Or was this "coincidence"?
    We would like some feedback on the flaws in our algorithm :)
  """

  position = currentGameState.getPacmanPosition()
  foodGrid = currentGameState.getFood()
  foodList = currentGameState.getFood().asList()
  ghostStates = currentGameState.getGhostStates()
  ghostDistance = [manhattanDistance(ghostState.getPosition(), position) for ghostState in ghostStates]
  scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
  wallList = currentGameState.getWalls()

  # Help Functions

  def minFoodDist(foodList, position, wallList):
    """
        Returns the minimum food distance
    """
    distances = [manhattanDistance(food, position) for food in foodList]
    minDistance = min(distances)
    indeces = [i for i, j in enumerate(distances) if j == minDistance]

    width = wallList.width - 1
    height = wallList.height - 1

    xFood = indeces[0] % width
    yFood = int(indeces[0]/width)

    xPac, yPac = position

    if(xFood == xPac):
        if (yPac > yFood):
            if wallList[xPac][yPac-1]:
                minDistance += 4
        else:
            if wallList[xPac][yPac+1]:
                minDistance += 4
    if(yFood == yPac):
        if (xPac > xFood):
            if wallList[xPac-1][yPac]:
                minDistance += 4
        else:
            if wallList[xPac+1][yPac]:
                minDistance += 4

    #print(indeces)
    #print(distance)
    return minDistance

  def foodCount(foodList):
    """
        Returns the amount of remaining food
    """
    sum = 0
    for food in foodList:
        if food:
            sum += 1
    return sum

  if currentGameState.isWin() or foodCount(foodList) == 0:
      #print("WIN")
      return 500000
  if currentGameState.isLose():
      return -500000

  heuristic = 10000

  heuristic += -foodCount(foodList) * 100
  if(sum(scaredTimes) > 2):
        heuristic += -minFoodDist(foodList, position, wallList)
        heuristic += 20
  else:
        heuristic += -minFoodDist(foodList, position, wallList)
        if(min(ghostDistance) < 3):
            heuristic += min(ghostDistance) * 10
        else:
            heuristic += 20

  # print("Heuristic= -minFoodDistance: ", minFoodDist(foodList, position), "=", heuristic)
  return heuristic

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest (not used in this course)
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

