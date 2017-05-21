"""
epsilon_greedy.py
Janbaanz Launde
Apr 2, 2017
"""
from policy import Policy
import random


class EpsilonGreedy(Policy):
    """Standard epsilon greedy, linear regret."""
    def __init__(self, contexts, epsilon=.1):
        super(EpsilonGreedy, self).__init__(contexts)
        self.name = 'EpsilonGreedy({})'.format(epsilon)
        self.ep = epsilon
        self.total_arms, self.d = contexts.shape
        self.arm_rewards = [0 for i in range(self.total_arms)]

    def predict_arm(self, contexts=None):
        rewards = [(i, self.arm_rewards[i]) for i in range(self.total_arms)]
        if random.random() >= self.ep:
            # exploitation step
            return max(rewards, key=lambda x: x[1])[0]
        else:
            # exploration step
            return random.choice(range(self.total_arms))

    def pull_arm(self, arm, reward, contexts=None):
        self.arm_rewards[arm] += reward
