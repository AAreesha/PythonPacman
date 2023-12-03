from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class SarsaLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.QValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.QValues[(state, action)]

        util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        qvalues = [self.getQValue(state, action) for action in self.getLegalActions(state)]
        if not len(qvalues): return 0.0
        return max(qvalues)
        #util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if not len(legalActions):
          return None
        QValue = -1e10
        for legalAction in legalActions:
          QValueTemp = self.getQValue(state, legalAction)
          if QValueTemp > QValue:
            action = legalAction
            QValue = QValueTemp
        return action
        #util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if not len(legalActions):
          return action
        randomAction = util.flipCoin(self.epsilon)
        if randomAction:
          action = random.choice(legalActions)
        else:
          action = self.getPolicy(state)
        return action

        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        #legalActions = self.getLegalActions(state)
        
        nextAction = self.getAction(nextState)
        curQValue = self.getQValue(state, action)
        nextQValue = self.getQValue(nextState, nextAction)
        #if not legalActions:
        #  self.QValues[(state, action)] = (1-self.alpha) * curQValue + self.alpha * reward
        #else:
        self.QValues[(state, action)] = (1-self.alpha) * curQValue + self.alpha * (reward \
                                          + self.discount * nextQValue )
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class SarsaAgent(SarsaLearningAgent):
    "Exactly the same as SarsaLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """

        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        SarsaLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = SarsaLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


