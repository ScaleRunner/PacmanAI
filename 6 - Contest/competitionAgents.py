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

  def __init__(self, index=0):
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

    # Access to the graphics
    self.display = None

    # useful function to find functions you've defined elsewhere..
    # self.usefulFunction = util.lookup(usefulFn, globals())
    # self.evaluationFunction = util.lookup(evalFn, globals())

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

    # Static world properties
    self.wallList = gameState.getWalls()
    self.wallHeight = self.wallList.height
    self.wallWidth = self.wallList.width

    # Determine in which world you are
    if self.wallHeight == 9 and self.wallWidth == 25:
        self.world = 'level0'
    if self.wallHeight == 7 and self.wallWidth == 20:
        self.world = 'level1'
    if self.wallHeight == 13 and self.wallWidth == 20:
        self.world = 'level2'
    if self.wallHeight == 27 and self.wallWidth == 28:
        self.world = 'level3'
    else:
        self.world = 'unknown'

    # Set the depth at which you want to search
    if self.world == 'level0':
        self.depth = 2
        self.timeForComputing = .2
    if self.world == 'level1':
        self.depth = 3
        self.timeForComputing = .2
    if self.world == 'level2':
        self.depth = 2
        self.timeForComputing = .3
        self.capsuleImpulse = True
    if self.world == 'level3':
        self.depth = 3
        self.timeForComputing = .25
    if self.world == 'unknown':
        self.depth = 2
        self.timeForComputing = .2

    # Prepare for the pacman ExploredList
    self.exploredListGrid = [[0 for x in range(100)] for x in range(100)]
    self.exploredList = []

    # Prepare for the ghost properties
    # ghostIndex, DistanceToGhost, ScaredTime = ghost
    self.ghosts = [(0, float('Inf'), 0), (1, float('Inf'), 0), (2, float('Inf'), 0), (3, float('Inf'), 0)]

    # If the response is triggered to get a capsule, than go get it
    self.capsuleImpulse = False

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

class MyPacmanAgent(CompetitionAgent):
  """
  This is going to be your brilliant competition agent.
  You might want to copy code from BaselineAgent (above) and/or any previos assignment.
  """

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.
    Just like in the previous projects, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}. 
    """
    # Add current position to your exploredList (only look at the last 20 positions)
    x, y = gameState.getPacmanPosition()
    self.exploredListGrid[x][y] += 1
    self.exploredList.append((x, y))
    if len(self.exploredList) > 20:
        x, y = self.exploredList.pop(0)
        self.exploredListGrid[x][y] += -1

    # Update the previous food and capsule state
    self.foodGrid = gameState.getFood()
    self.capsules = gameState.getCapsules()
    self.oldScore = gameState.getScore()
    self.nrOfFoods = len(self.foodGrid.asList())

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
    if maxAction == '':
        if self.lastAction in pacmanActions:
            return self.lastAction
        else:
            import random
            return random.choice(pacmanActions)
    self.lastAction = maxAction
    return maxAction

  def evaluationFunction(self, state):
    """
    Masterful Evaluation Function
    """
    # Utilise a counter for the heuristic
    heuristic = util.Counter()

    # World Properties
    oldFoodGrid = self.foodGrid
    foodGrid = state.getFood()
    nrOfFoods = len(foodGrid.asList())
    capsules = self.capsules

    # Pacman Properties
    pacmanPosition = state.getPacmanPosition()
    xPacman, yPacman = pacmanPosition
    pacmanActions = set(Actions.getLegalNeighbors(pacmanPosition, self.wallList))

    # Ghost Properties
    ghostPositions = state.getGhostPositions()
    ghostStates = state.getGhostStates()
    nrGhosts = state.getNumAgents() - 1
    ghostActions = []

    totalGhostDistance = 0
    minGhostDistance = float('Inf')
    minScaredGhostDistance = float('Inf')
    maxScaredTimer = float('-Inf')

    for ghost in range(nrGhosts):
        ghostIndex, ghostDistance, scaredTime= self.ghosts[ghost]
        ghostDistance = self.getMazeDistance(pacmanPosition, ghostPositions[ghost])
        totalGhostDistance += ghostDistance
        scaredTime = ghostStates[ghost].scaredTimer
        ghostActions += Actions.getLegalNeighbors(ghostPositions[ghost], self.wallList)

        if ghostDistance < minScaredGhostDistance and scaredTime > 0:
            minScaredGhostDistance = ghostDistance

        if ghostDistance < minGhostDistance:
            minGhostDistance = ghostDistance

        if scaredTime > maxScaredTimer:
            maxScaredTimer = scaredTime

        self.ghosts[ghost] = (ghostIndex, ghostDistance, scaredTime)

    # Help Functions
    def minFoodDist(foodGrid, position):
        """
            Returns the minimum food distance
            It first searches for foods that are close by to save computation time.
        """
        x, y = position
        distances = []
        if (x < 7):
            x = 4
        if (x >= self.wallWidth - 2):
            x += -4
        if (y < 7):
            y = 4
        if (y >= self.wallHeight - 2):
            y += -4
        for xFood in range(x-3,x+3,1):
            for yFood in range (y-3,y+3,1):
                food = foodGrid[xFood][yFood]
                if food:
                    distances.append(self.getMazeDistance((xFood, yFood), position))
        if len(distances) == 0:
            distances = [self.getMazeDistance(food, position) for food in foodGrid.asList()]

        if len(distances) > 0:
            minDistance = min(distances)
            return minDistance
        else:
            return 0

    # Check for trapped situations (there are no good options for pacman)
    goodActions = pacmanActions - set(ghostActions)
    if not goodActions:
        heuristic['trapped'] = -2000

    # Lose case
    if state.isLose():
        return float('-Inf')

    # Prefer not to visit already visited places (avoiding loops)
    if self.exploredListGrid[xPacman][yPacman] > 2 and not(maxScaredTimer > 0):
        heuristic['beenThere'] = -100 * self.exploredListGrid[xPacman][yPacman]
        foodDifference = self.nrOfFoods - nrOfFoods
        if foodDifference == 1:
            heuristic['OneFoodLess'] = 1000

    # Minimum distance to the food
    if not(maxScaredTimer > 0):
        if not oldFoodGrid[xPacman][yPacman]:
            heuristic['minFoodDistance'] = -minFoodDist(foodGrid, pacmanPosition)/(self.wallWidth * self.wallHeight)

    # Eating ghosts
    if maxScaredTimer > 1:
    # if maxScaredTimer < 2 * minScaredGhostDistance and maxScaredTimer > 0:
        heuristic['nearScaredGhost'] = 100 / minScaredGhostDistance
    # Prioritise ghost eating when ghosts are scared, not food
    if maxScaredTimer > 0:
        if oldFoodGrid[xPacman][yPacman]:
            heuristic['eatFood'] = -10

    # Capsule Reasoning
    capsuleDistance = [self.getMazeDistance(capsule, pacmanPosition) for capsule in capsules]
    if capsuleDistance and minGhostDistance < 10 and min(capsuleDistance) < 10:
        self.capsuleImpulse = True
    # Eat the powerpelets before finishing the level
    if capsuleDistance and self.nrOfFoods == 1 and oldFoodGrid[xPacman][yPacman]:
        heuristic['PowerpeletFirst'] = -1000
        self.capsuleImpulse = True
    # If Ghosts not scared, than don't give higher heuristic for capsule eating
    if self.capsuleImpulse and not(maxScaredTimer > 0):
        if capsuleDistance:
            heuristic['nearCapsule'] = 10 / min(capsuleDistance)
        if pacmanPosition in capsules:
            heuristic['eatCapsule'] = 300
            self.capsuleImpulse = False

    # World specific heuristics
    if self.world == 'level0' or self.world == 'level1':
        if self.nrOfFoods == 1 and maxScaredTimer > 0 and oldFoodGrid[xPacman][yPacman]:
            heuristic['GhostsFirst'] = -10000

    heuristic['score'] = state.getScore()

    return heuristic.totalCount()

MyPacmanAgent = MyPacmanAgent