"""
linUCB.py
Janbaanz Launde
Apr 2, 2017
"""
from policy import Policy
from math import sqrt
import numpy as np


class LinUCB(Policy):
    """
    LinUCB, as per Li, Chu, Langford, Schapire 2012
    """
    def __init__(self, contexts, alpha=.01):
        super(LinUCB, self).__init__(contexts)
        self.name = 'LinUCB'
        self.t = 1
        # self.d = dim**2
        self.total_arms, self.d = contexts.shape
        self.A = np.identity(self.d)
        self.b = np.zeros(self.d)
        self.pulls = [0 for i in range(self.total_arms)]
        self.arm_prop = {}
        self.alpha = alpha

    def predict_arm(self, contexts):
        alpha = self.alpha
        beta_hat = np.dot(np.linalg.inv(self.A), self.b)
        ps = [0 for i in range(self.total_arms)]

        for arm in range(self.total_arms):
            z = contexts[arm].flatten()
            x = z

            if self.pulls[arm] == 0:  # initialize
                self.arm_prop[arm] = {'A': np.identity(self.d),
                                      'B': np.zeros((self.d, self.d)),
                                      'b': np.zeros(self.d)}

            a_var = self.arm_prop[arm]

            theta_hat = np.dot(np.linalg.inv(a_var['A']),
                               a_var['b'] -
                               np.dot(a_var['B'], beta_hat))

            A0inv = np.linalg.inv(self.A)
            zTA0inv = np.dot(z, A0inv)
            aInv = np.linalg.inv(a_var['A'])
            xTAinv = np.dot(x, aInv)
            s = abs((np.dot(zTA0inv, z) -
                 2 * np.dot(np.dot(np.dot(zTA0inv, a_var['B']),
                                   aInv), x) +
                 np.dot(xTAinv, x) +
                 np.dot(np.dot(np.dot(np.dot(np.dot(xTAinv,
                                                    a_var['B']), A0inv),
                                      a_var['B'].T), aInv), x)))
            # print s, alpha, np.inner(x, theta_hat), np.dot(z, beta_hat)
            ps[arm] = (arm, (np.dot(z, beta_hat) + np.inner(x, theta_hat) +
                       alpha * sqrt(s)))
            # print ps[arm]

        return max(ps, key=lambda x: x[1])[0]

    def pull_arm(self, arm, reward, contexts):
        self.pulls[arm] += 1
        z = contexts[arm].flatten()
        x = z
        r = reward
        B = self.arm_prop[arm]['B']
        aInv = np.linalg.inv(self.arm_prop[arm]['A'])
        b = self.arm_prop[arm]['b']

        self.A += np.dot(np.dot(B.T, aInv), B)
        self.b += np.dot(np.dot(B.T, aInv), b)
        self.arm_prop[arm]['A'] += np.outer(x, x)
        self.arm_prop[arm]['B'] += np.outer(x, z)
        self.arm_prop[arm]['b'] += r * x

        aInv = np.linalg.inv(self.arm_prop[arm]['A'])
        B = self.arm_prop[arm]['B']
        b = self.arm_prop[arm]['b']

        self.A += np.outer(z, z) - np.dot(np.dot(B.T, aInv), B)
        self.b += r * z - np.dot(np.dot(B.T, aInv), b)

        self.t += 1
