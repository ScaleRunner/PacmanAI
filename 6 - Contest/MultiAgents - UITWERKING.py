# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
from game import Actions
import random, util

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

    Note: ideally this score is an estimate of the final score pacman can achieve
    by moving through this state under the assumption of optimal play by both pacman 
    and the ghosts. That is, essentially this is a heuristic function for multi-player games.
    (However, in reality the actual score doens't matter, so long as states which would realise
     higher final scores have higher evaluated values.)    

    """

    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # Successor state is a win state
    if successorGameState.isWin():
      return float('inf')

    score = 0
    # Check for collissions with ghosts
    # High positive score if Pacman can easily eat a ghost (i.e. one very nearby)
    # High negative score if Pacman can easily be eaten by ghost (i.e. one very nearby)
    ghostPositions = successorGameState.getGhostPositions()
    for index, ghostPosition in enumerate(ghostPositions):
      ghostScaredTime = newScaredTimes[index]
      distanceToGhost = util.manhattanDistance(newPos, ghostPosition)
      if ghostScaredTime > 0:
        # Pacman can eat the ghost
        if distanceToGhost < 1:
          score = float('inf')
          break
        else:
          # Pacman shouldn't risk it
          if distanceToGhost < 2:
            score = float('-inf')
            break

    # Calculate food distance over all food
    # N.B. better would be to use the minimum spanning tree distance
    foodDistance = 0
    for food in newFood:
      foodDistance += util.manhattanDistance(newPos, food)
    # N.B. high foodDistance is bad, so we invert (1/x) it so high distances score low
    #  using 10/foodDistance means we score 1 for having all the food within 10 steps of pacman
    score = score - foodDistance;

    # Calculate ghost distance over all ghosts
    ghostDistance = 0
    for ghost in newGhostStates:
      ghostDistance += util.manhattanDistance(newPos, ghost.getPosition())
    # N.B. high ghostDistance is good, so don't invert that.  *But* nearby ghosts are bad so scale strongly      
    #   mult by 30 means we score and additional 30 for every unit a ghost is further away
    #   this is equivalent to saying that its worth taking 30 steps for each food pellet to avoid a ghost
    score += 30 * ghostDistance

    # N.B. low number food still to eat is good, so invert so low food to go scores high
    score = score - newFood.count()
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

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
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
    numAgents = 2

    # loop over possible actions for pacman evaluating the minimax score of each
    actions = gameState.getLegalActions(0)
    scores = []
    for action in actions:
      nextState = gameState.generateSuccessor(0, action)
      scores.append(self.minimax(nextState, (self.depth * numAgents - 1)))

    # identify the best action, and break ties by randomally picking one of them
    bestScore = max(scores)
    # list comprehension to identify all actions with the score == bestScore
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    return actions[chosenIndex]

  def minimax(self, state, depth):
    """ This function computes the minimax value of state as estimated by a search to the 
        given depth, where leaf nodes are scored by calling evaluation function"""
    # Terminate the depth-first style search when we reach the indicated depth bound
    # or of the state is a win/loss position
    if depth <= 0 or state.isWin() or state.isLose():
      return self.evaluationFunction(state)

    # N.B. this code works automatically for any number of ghosts.
    numAgents = 2
    # N.B. Pacman is always agent #0, ghosts are numbered after that
    agentIndex = (-depth) % numAgents
      
    # initial values for this state are 'worse case' values for this agent,
    # i.e. -inf for pacman (MAX) or inf for ghost (MIN)
    if (agentIndex == 0):
      score = float('-inf') 
    else: 
      score = float('inf')
    for action in state.getLegalActions(agentIndex):
      guess = self.minimax(state.generateSuccessor(agentIndex, action), depth - 1)
      if (agentIndex == 0):
        score = max(score,guess) # MAX agent (pacman) picks the best successor state
      else: 
        score = min(score,guess) # MIN agent (ghost) picks the worst successor state
      #N.B. a shorter 'python' way to write the above if is
      #  score = max(score,guess) if (agentIdx==0) else min(score,guess)        
    return score

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # N.B. this code works automatically for any number of ghosts.
    numAgents = 2

    alpha = float('-inf')
    beta = float('inf')
    actions = gameState.getLegalActions(0)
    scores = []
    for action in actions:
      nextState = gameState.generateSuccessor(0, action)
      score = self.alphabeta(nextState, (self.depth * numAgents - 1), alpha, beta)
      scores.append(score)
      alpha = max(score, alpha)

    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    return actions[chosenIndex]

  def alphabeta(self, state, depth, alpha, beta):
    """ This function computes the minimax value of state as estimated by a search to the 
        given depth where leaf nodes are scored by calling evaluation function.
        This version uses alpha-beta pruning to reduce the number of nodes which must
        be evaulated. 
    	 beta represents MIN (non-pacman) player best choice, i.e. lower bound on value of current choice
           MIN won't allow us to use this state if alpha can choose better value.
    	 alpha represents MAX (i.e. pacman) upper-bound on our best choice 
    	   MAX won't pick one of these options if it's worse than best value in different branch
         """
    # Terminate the depth-first style search when we reach the indicated depth bound
    # or of the state is a win/loss position
    if depth <= 0 or state.isWin() or state.isLose():
      return self.evaluationFunction(state)

    
    numAgents = 2
    # N.B. Pacman is always agent #0
    agentIndex = (-depth) % numAgents

    # initial values for this state are 'worse case' values for this agent,
    # i.e. -inf for pacman (MAX) or inf for ghost (MIN)
    for action in state.getLegalActions(agentIndex):
      guess = self.alphabeta(state.generateSuccessor(agentIndex, action), depth - 1, alpha, beta)
      # pacman agent = MAX = alpha
      if agentIndex == 0:
        alpha = max(alpha, guess) # update the alpha value
        if beta <= alpha:
          break # stop if alpha is bigger than beta, i.e. MIN can choose other branch with value==beta
      else: # ghost = MIN = beta
        beta = min(beta, guess) # update beta value
        if beta <= alpha:
          break # stop if alpha is bigger than beta, i.e. MAX can choose other branch with value==alpha

    return alpha if (agentIndex == 0) else beta


class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
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
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  food = currentGameState.getFood()
  walls = currentGameState.getWalls()
  ghosts = currentGameState.getGhostPositions()
  ghostStates = currentGameState.getGhostStates()
  capsules = currentGameState.getCapsules()

  nextx, nexty = currentGameState.getPacmanPosition()

  features = util.Counter()

  # Calculate the number of ghosts that are one step away
  for ghost in ghosts:
    if (nextx, nexty) in Actions.getLegalNeighbors(ghost, walls):
      features['numGhostsNear'] += 1

  # Check for food at location
  if not features['numGhostsNear'] and food[nextx][nexty]:
    features['foodAtLocation'] = 1.0

  # Win/lose
  if currentGameState.isLose():
    features['lose'] = -100
  if currentGameState.isWin():
    features['win'] = 100

  # Can pacman eat a ghost
  scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
  minTimeLeft = min(scaredTimes)
  if minTimeLeft > 1:
    features['eatGhost'] = 1/10.0
    features['numGhostsNear'] = 0

  # Reason about the capsules
  ghostDistances = [util.manhattanDistance(ghost, (nextx, nexty)) for ghost in ghosts]
  minGhostDistance = min(ghostDistances)
  capsuleDistances = [util.manhattanDistance(capsule, (nextx, nexty)) for capsule in capsules]
  if capsuleDistances:
    minCapsuleDistance = min(capsuleDistances)
    features['closestCapsule'] = (-1.0*minCapsuleDistance)/(walls.width * walls.height)
    if minGhostDistance < 3 and (nextx, nexty) in capsules:
      features['eatCapsule'] = 5.0
  
  # Check if in next situation pacman is trapped
  actionsPossible = set(Actions.getLegalNeighbors((nextx, nexty), walls))
  possibleGhostPositions = []
  for ghost in ghosts:
    possibleGhostPositions += Actions.getLegalNeighbors(ghost, walls)
  actionsRemaining = actionsPossible - set(possibleGhostPositions)
  if not actionsRemaining:
    features['trapped'] = 1/10.0

  # Food left
  features['numFoodLeft'] = (-20.0*len(food.asList()))/(walls.width * walls.height)

  return features.totalCount()
  

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
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

