# 4455770       Dennis Verheijden       KI
# 4474139       Remco van der Heijden   KI

# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
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
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """

     util.raiseNotDefined()
           

def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first [p 85].
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm [Fig. 3.7].
  
  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:

  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())
  """

  path = util.Stack() # path bevat: [(huidige positie, acties tot nu toe) ...]
  exploredList = []
  start = problem.getStartState()
  path.push((start, [])) # Er zijn nog geen mogelijke acties
  #print("Startstate:", start)

  while not path.isEmpty():
      position, actions = path.pop() #opsplitsen van een element in path
      for next, direction, cost in problem.getSuccessors(position):
          if not next in exploredList: #loop-detectie
              #print("Volgende node:", next, "in", direction)
              if problem.isGoalState(next):
                  return actions + [direction]
              path.push((next, actions + [direction]))
              exploredList.append(next)

  return[]

def breadthFirstSearch(problem):

  path = util.Queue() #path bevat: [(huidige positie, acties tot nu toe) ...]
  exploredList = []
  start = problem.getStartState()
  path.push((start, [])) #Er zijn nog geen mogelijke acties
  #print("Startstate:", start)

  while not path.isEmpty():
      position, actions= path.pop() #opsplitsen van een element in path
      for next, direction, cost in problem.getSuccessors(position):
          if not next in exploredList: #loop-detectie
              #print("Volgende node:", next, "in", direction)
              if problem.isGoalState(next):
                  return actions + [direction]
              path.push((next, actions + [direction]))
              exploredList.append(next)

  return[]

def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  path = util.PriorityQueue() #path bevat: [(huidige positie, acties tot nu toe) ...]
  exploredList= []
  start = problem.getStartState()
  path.push((start, []),0) #Er zijn nog geen mogelijke acties
  #print("Startstate:", start)

  while not path.isEmpty():
      position, actions = path.pop() #opsplitsen van een element in path
      for next, direction, cost in problem.getSuccessors(position):
          if not next in exploredList: #loop-detectie
              #print("Volgende node:", next, "in", direction)
              if problem.isGoalState(next):
                  return actions + [direction]
              new_actions = actions + [direction]
              path.push((next, actions + [direction]),problem.getCostOfActions(new_actions))
              exploredList.append(next)

  return[]

def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  path = util.PriorityQueue() #path bevat: [(huidige positie, acties tot nu toe) ...]
  exploredList = []
  start = problem.getStartState()
  path.push((start, []),0) #Er zijn nog geen mogelijke acties
  #print("Startstate:", start)

  while not path.isEmpty():
      position, actions = path.pop() #opsplitsen van een element in path
      for next, direction, cost in problem.getSuccessors(position):
          if not next in exploredList: #loop-detectie
              #print("Volgende node:", next, "in", direction)
              if problem.isGoalState(next):
                  return actions + [direction]
              new_actions = actions + [direction]
              totalCost = problem.getCostOfActions(new_actions) + heuristic(next, problem)
              path.push((next, actions + [direction]),totalCost)
              exploredList.append(next)

  return[]

  "Bonus assignment: Adjust the getSuccessors() method in CrossroadSearchAgent class"
  "in searchAgents.py and test with:"
  "python pacman.py -l bigMaze -z .5 -p CrossroadSearchAgent -a fn=astar,heuristic=manhattanHeuristic "
  
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
