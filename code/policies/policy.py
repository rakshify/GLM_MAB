"""
policy.py
Janbaanz Launde
Apr 1, 2017
"""


class Policy(object):
    """Abstract class for all policies"""
    name = 'POLICY'

    def __init__(self, contexts):
        self.contexts = contexts

    def predict_arm(self, contexts=None):
        raise NotImplementedError("You need to override this function in child class.")

    def pull_arm(self, arm, reward, contexts=None):
        raise NotImplementedError("You need to override this function in child class.")
