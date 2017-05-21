"""
glmUCB.py
Janbaanz Launde
Apr 1, 2017
"""
from policy import Policy
from math import exp, log, sqrt
import numpy as np
import scipy.optimize


class GLMUCB(Policy):
    """
        GLM-UCB algorithm given by Filippi et al. in ICML-2013. Parametric Bandits.
        Default link function is logistic (exp(x) / (1+exp(x)))
    """
    def __init__(self, contexts):
        super(GLMUCB, self).__init__(contexts)
        self.name = 'GLM-UCB'
        self.t = 1
        self.total_arms, self.d = contexts.shape
        self.M_ = np.eye(self.d)
        self.M_inv_ = np.eye(self.d)
        self.rewards = []
        self.context = []
        self.pulls = [0 for i in range(self.total_arms)]                         # {arm: 0 for arm in contexts}
        self.prev = np.zeros(self.d)

    def predict_arm(self, contexts):
        """
            FUNCTION TO GET THE ARM WITH BEST REWARD. ALGORITHM USED- GLMUCB

            Parameters
            ----------
            contexts:   numpy matrix, shape(n_arms, n_features)
                        matrix of all the context features passed with one rows
                        as the feature of one arm

            Returns
            -------
            choice: int, the index of arm with the best reward
        """
        for i in range(self.total_arms):  # initialize
            if self.pulls[i] == 0:
                return i

        theta_hat = scipy.optimize.root(self.__to_optimize,
                                        self.prev).x
        est_rew = [(a, self.acquisition_function(theta_hat, contexts[a]))
                   for a in range(self.total_arms)]

        self.prev = theta_hat
        return max(est_rew, key=lambda x: x[1])[0]

    def pull_arm(self, arm, reward, contexts):
        """
            FUNCTION TO PULL THE ARM WITH BEST REWARD AND UPDATE DESIGN MATRIX.

            Parameters
            ----------
            arm:        int, Index of the best arm
            reward:     int, binary 0/1 reward, whether the recommended article
                        was clicked or not
            contexts:   numpy matrix, shape(n_arms, n_features)
                        matrix of all the context features passed with one rows
                        as the feature of one arm

            Returns
            -------
            void
        """
        X = contexts[arm].flatten()
        self.rewards.append(reward)
        self.context.append(X)
        self.M_ += np.outer(X, X)
        self.M_inv_ = np.linalg.inv(self.M_)
        self.pulls[arm] += 1
        self.t += 1

    def acquisition_function(self, theta, context):
        """
            ACQUISITION FUNCTION FOR UCB TO BALANCE ECPLOITATION & EXPLORATION.

            Parameters
            ----------
            theta:      ndarray, shape(1, n_features)
                        predictor for exploitation part
            context:    ndarray, shape(1, n_features)
                        context feature of the arm for which value of acquisition
                        function is to be found out

            Returns
            -------
            value:      float, reward on which basis the algo predicts best arm
        """
        X = np.array(context).flatten()
        t = len(self.rewards)

        exploit = self.link(np.inner(X, theta))
        # As per filippi, using ro value for exploration part to be sqrt(3 * log(t))
        explore = sqrt(3 * log(t)) * sqrt(np.dot(np.dot(X.T, self.M_inv_), X))

        return exploit + explore

    def __to_optimize(self, theta):
        """
            FUNCTION TO GET EQUATION "6" FROM FILIPPI et. al.

            Parameters
            ----------
            theta:      ndarray, shape(1, n_features)
                        predictor, the variable for equation whose root will
                        give us theta_hat

            Returns
            -------
            value:      float, reward on which basis the algo predicts best arm
        """
        to_sum = []
        for t in range(len(self.rewards)):
            R_k = self.rewards[t]
            mu_k = self.link(np.inner(self.context[t], theta))
            m_ak = self.context[t]
            # Equation "6" to be solved as per filippi:-
            # sum ((R_k - mu_k) * xk) = 0
            to_sum.append((R_k - mu_k) * m_ak)

        return np.sum(to_sum, 0)

    def link(self, x):
        try:
            return 1 / (1 + exp(-x))
        except:
            return exp(x) / (1 + exp(x))
