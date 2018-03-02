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


from util import manhattanDistance
from game import Directions
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
        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        # if the successor game state is win state, return the maximum score 
        if successorGameState.isWin():
          return float("inf")
        
        WEIGHT_GHOST = 20
        WEIGHT_CAPSULE = 50
        WEIGHT_FOOD = 10
        # calculate the manhattan distance between pacman and ghost, then evaluate score
        for ghost in newGhostStates:
          d = manhattanDistance(ghost.getPosition(), newPos)
          if d == 0:
            if ghost.scaredTimer != 0:
              score += WEIGHT_GHOST
            else:
              score -= WEIGHT_GHOST
          else:
            score -= WEIGHT_GHOST/d
        # calculate food's effect
        disToFood = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if disToFood:
          score += WEIGHT_FOOD / min(disToFood)
        # calculate capsule's effect
        for capsule in currentGameState.getCapsules():
          d = manhattanDistance(capsule, newPos)
          if d == 0:
            score += WEIGHT_CAPSULE
          else:
            score += WEIGHT_FOOD/d
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
    def isGameOver(self, state, depth):
      """
        check whether the current game state is terminal state
      """
      return depth == 0 or state.isWin() or state.isLose()
    
    def minimax(self, state):
      _, move = self.max_play(state,self.depth)
      return move

    def max_play(self,state,depth):
        if self.isGameOver(state, depth):
          return self.evaluationFunction(state), Directions.STOP
        actions = state.getLegalActions()
        maxScore = float('-inf')
        maxIndex = 0
        for index, score in enumerate([self.min_play(state.generateSuccessor(0, ac), depth, 1) for ac in actions]):
          if score > maxScore:
            maxScore = score
            maxIndex = index
        return maxScore, actions[maxIndex]

    def min_play(self, state, depth, agent):
        if self.isGameOver(state, depth):
          return self.evaluationFunction(state), Directions.STOP
        actions = state.getLegalActions(agent)
        scores=[]
        if agent == state.getNumAgents()-1:
          scores = [self.max_play(state.generateSuccessor(agent,ac), depth - 1)[0] for ac in actions]
        else:
          scores = [self.min_play(state.generateSuccessor(agent,ac), depth, agent + 1) for ac in actions]
        minScore = min(scores)
        minIndex = 0
        for index, score in enumerate(scores):
          if score == minScore:
            minIndex = index
        return minScore, actions[minIndex]

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
        return self.minimax(gameState)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def isGameOver(self, state, depth):
      return depth == self.depth or state.isWin() or state.isLose()

    def minimax(self, state, depth, agent, a=float('-inf'), b=float('inf')):
      if agent == state.getNumAgents():
        depth += 1
        agent = 0

      if self.isGameOver(state, depth):
        return self.evaluationFunction(state), None
      return self.helper(state, depth, agent, a, b)

    def helper(self, state, depth, agent, a, b):
      bestAction = None
      v = float('-inf')
      if agent != 0:
        v = float('inf')
      for ac in state.getLegalActions(agent):
        ss = state.generateSuccessor(agent, ac)
        score,_ = self.minimax(ss, depth, agent+1, a, b)
        if agent == 0:
          v, bestAction = max((v, bestAction), (score, ac))
          if v > b:
            return v, bestAction
          a = max(a, v)
        else:
          v, bestAction = min((v, bestAction), (score, ac))
          if v < a:
            return v, bestAction
          b = min(b, v)
      return v, bestAction

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        _, action = self.minimax(gameState, 0, 0)
        return action
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def isGameOver(self, state, depth):
      return depth == self.depth or state.isWin() or state.isLose()

    def expectimax(self, state, depth, agent):
      if agent == state.getNumAgents():
        agent = 0
        depth += 1
      if self.isGameOver(state, depth):
        return self.evaluationFunction(state), Directions.STOP
      if agent == 0:
        return self.max_play(state, depth)
      else:
        return self.exp_play(state, depth, agent)
    
    def max_play(self, state, depth):
      bestAction = None
      v = float('-inf')
      actions = state.getLegalActions()
      for ac in actions:
        ss = state.generateSuccessor(0, ac)
        score, _ = self.expectimax(ss, depth, 1)
        v, bestAction = max((v, bestAction), (score, ac))
      return v, bestAction
    
    def exp_play(self, state, depth, agent):
      bestAction = None
      actions = state.getLegalActions(agent)
      scores = [self.expectimax(state.generateSuccessor(agent, ac), depth, agent+1)[0] for ac in actions]
      return sum(scores) / float(len(scores)), None


    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        _,action = self.expectimax(gameState, 0, 0)
        return action
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    pacmanPos = currentGameState.getPacmanPosition()
    foodPos = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()

    v = currentGameState.getScore()
    if currentGameState.isWin():
      return float('inf')
    WEIGHT_GHOST = 30.0
    WEIGHT_CAPSULE = 100.0
    WEIGHT_FOOD = 15.0
    # calculate the manhattan distance between pacman and each ghost, then evaluate score.
    for g in ghostStates:
      d = manhattanDistance(g.getPosition(), pacmanPos)
      if d == 0:
        if g.scaredTimer != 0:
          v += WEIGHT_GHOST
        else:
          v -= WEIGHT_GHOST
      else:
        v -= WEIGHT_GHOST/d
    # calculate food's effect by finding the closest food 
    foodDis = [manhattanDistance(pacmanPos, f) for f in foodPos.asList()]
    if foodDis:
      v += WEIGHT_FOOD/min(foodDis)
    # calculate capsule's effect
    for capsule in capsules:
      d = manhattanDistance(capsule, pacmanPos)
      if d == 0:
        v += WEIGHT_CAPSULE
      else:
        '''
        if capsule is not at pacman's postion, we will see capsule the same as food
        since we don't wanna pacman die on the way eating capsule
        '''
        v += WEIGHT_FOOD/d
    return v

# Abbreviation
better = betterEvaluationFunction

