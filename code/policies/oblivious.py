"""
oblivious.py
Janbaanz Launde
Apr 2, 2017
"""
from policy import Policy
import random


class Oblivious(Policy):
    """Picks arms entirely at random."""
    def __init__(self, contexts):
        self.name = 'Random'
        self.total_arms, self.d = contexts.shape

    def predict_arm(self, contexts=None):
        return random.choice(range(self.total_arms))

    def pull_arm(self, arm, reward, contexts=None):
        pass
