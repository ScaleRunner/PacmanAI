# 4455770       Dennis Verheijden       KI
# 4474139       Remco van der Heijden   KI

# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance, nearestPoint
from game import Directions, Agent, Actions
import random, util
import distanceCalculator

class CompetitionAgent(Agent):
  """
  A base class for competition agents.  The convenience methods herein handle
  some of the complications of the game.

  Recommended Usage:  Subclass CompetitionAgent and override getAction.
  """

  #############################
  # Methods to store key info #
  #############################

  def __init__(self, index=0, timeForComputing = .1, depth='2'):
    """
    Lists several variables you can query:
    self.index = index for this agent
    self.distancer = distance calculator (contest code provides this)
    self.timeForComputing = an amount of time to give each turn for computing maze distances
        (part of the provided distance calculator)
    """
    # Agent index for querying state, N.B. pacman is always agent 0
    self.index = index

    # Maze distance calculator
    self.distancer = None

    # Time to spend each turn on computing maze distances
    self.timeForComputing = timeForComputing

    # Access to the graphics
    self.display = None

    # Where pacman has been
    self.exploredList = [[0 for x in range(100)] for x in range(100)]

    # useful function to find functions you've defined elsewhere..
    # self.usefulFunction = util.lookup(usefulFn, globals())
    # self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)


  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields.
    
    A distanceCalculator instance caches the maze distances 
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)
    """
    self.distancer = distanceCalculator.Distancer(gameState.data.layout)
    
    # comment this out to forgo maze distance computation and use manhattan distances
    self.distancer.getMazeDistances()
    
    import __main__
    if '_display' in dir(__main__):
      self.display = __main__._display


  #################
  # Action Choice #
  #################

  def getAction(self, gameState):
    """
    Override this method to make a good agent. It should return a legal action within
    the time limit (otherwise a random legal action will be chosen for you).
    """
    util.raiseNotDefined()

  #######################
  # Convenience Methods #
  #######################

  def getFood(self, gameState):
    """
    Returns the food you're meant to eat. This is in the form of a matrix
    where m[x][y]=true if there is food you can eat (based on your team) in that square.
    """
    return gameState.getFood()

  def getCapsules(self, gameState):
    return gameState.getCapsules()


  def getScore(self, gameState):
    """
    Returns how much you are beating the other team by in the form of a number
    that is the difference between your score and the opponents score.  This number
    is negative if you're losing.
    """
    return gameState.getScore()

  def getMazeDistance(self, pos1, pos2):
    """
    Returns the distance between two points; These are calculated using the provided
    distancer object.

    If distancer.getMazeDistances() has been called, then maze distances are available.
    Otherwise, this just returns Manhattan distance.
    """
    d = self.distancer.getDistance(pos1, pos2)
    return d


class BaselineAgent(CompetitionAgent):
  """
    This is a baseline reflex agent to see if you can do better.
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
  """

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    
    # Collect legal moves and successor states
    x,y = gameState.getLegalActions()

    # try each of the actions and pick the best one
    scores=[]
    for action in legalMoves:
      successorGameState = gameState.generatePacmanSuccessor(action)
      scores.append(self.evaluationFunction(successorGameState))
    
    # get the best action
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"
    return legalMoves[chosenIndex]

  def evaluationFunction(self, state):
   # Useful information you can extract from a GameState (pacman.py)
   return state.getScore()
  
   
class TimeoutAgent( Agent ):
  """
  A random agent that takes too much time. Taking
  too much time results in penalties and random moves.
  """
  def __init__( self, index=0 ):
    self.index = index
    
  def getAction( self, state ):
    import random, time
    time.sleep(2.0)
    return random.choice( state.getLegalActions( self.index ) )
	

class MyPacmanAgent(CompetitionAgent):
  """
  This is going to be your brilliant competition agent.
  You might want to copy code from BaselineAgent (above) and/or any previos assignment.
  """

  # The following functions have been declared for you,
  # but they don't do anything yet (getAction), or work very poorly (evaluationFunction)

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.
    Just like in the previous projects, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}. 
    """
    # Add current position to your exploredList
    x, y = gameState.getPacmanPosition()
    self.exploredList[x][y] += 1

    self.foodGrid = gameState.getFood()
    self.capsules = gameState.getCapsules()

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
    if maxAction == '':
        return "Stop"
    return maxAction

  def evaluationFunction(self, state):
    """
    Masterful Evaluation Function
    """ 
    position = state.getPacmanPosition()
    x, y = position

    oldFoodGrid = self.foodGrid
    foodGrid = state.getFood()
    foodList = foodGrid.asList()
    ghostStates = state.getGhostStates()
    ghostPos = state.getGhostPositions()
    ghostDistance = [self.getMazeDistance(ghostState.getPosition(), position) for ghostState in ghostStates]
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    wallList = state.getWalls()
    heuristic = util.Counter()
    capsules = self.capsules

    # Help Functions
    def minFoodDist(foodList, position, wallList):
        """
            Returns the minimum food distance
        """
        distances = [self.getMazeDistance(food, position) for food in foodList]
        if len(distances) > 0:
            minDistance = min(distances)
            return minDistance
        else:
            return 0

    def foodCount(foodList):
        """
        Returns the amount of remaining food
        """
        sum = 0
        for food in foodList:
            if food:
                sum += 1
        return sum

    # Win or Lose
    if state.isWin():
        return float('Inf')
    if state.isLose():
        return float('-Inf')

    # Prefer not to visit already visited places
    heuristic['beenThere'] = -.5*self.exploredList[x][y]

    # Check for trapped situations
    pacmanActions = set(Actions.getLegalNeighbors(position,wallList))
    ghostActions = []
    for ghost in ghostPos:
        ghostActions += Actions.getLegalNeighbors(ghost, wallList)
    goodActions = pacmanActions - set(ghostActions)
    if not goodActions:
        heuristic['trapped'] = -2000

    # Minimum distance to the food
    if not oldFoodGrid[x][y]:
        heuristic['minFoodDistance'] = -minFoodDist(foodList,position,wallList)/(wallList.width * wallList.height)

    # Amount of food left
    heuristic['foodCount'] = -10*foodCount(foodList)/(wallList.width * wallList.height)

    # Ghosts one step away
    numGhostsNear = 0
    for ghost in ghostPos:
        if (x, y) in Actions.getLegalNeighbors(ghost, wallList):
            numGhostsNear += 1

    # safe supper
    if numGhostsNear == 0 and oldFoodGrid[x][y]:
        heuristic['safeFood'] = 10

    # Eating ghosts
    if min(scaredTimes) > 1 and min(ghostDistance) < 4:
        heuristic['nearScaredGhost'] = 10/min(ghostDistance)
        if position in ghostPos:
            print("IMA EATIN YA ASS")
            heuristic['eatGhost'] = 10

    # Capsule Reasoning
    capsuleDistance = [self.getMazeDistance(capsule, position) for capsule in capsules]
    if capsuleDistance and min(ghostDistance) < 3 and min(capsuleDistance) < 3:
        if min(capsuleDistance) > 0:
            heuristic['nearCapsule'] = 10/min(capsuleDistance)
        else:
            heuristic['eatCapsule'] = 15

    heuristic['score'] = state.getScore()

    # print("Heuristic= -minFoodDistance: ", minFoodDist(foodList, position), "=", heuristic)
    return heuristic.totalCount()

MyPacmanAgent=MyPacmanAgent