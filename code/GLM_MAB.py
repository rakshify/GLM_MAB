import numpy as np, inspect, helper, sys
from math import fabs, log, sqrt
from sklearn import linear_model as LM
from time import time

class GLM_MAB:
	"""Multi Arm Bandit Problem in Generalised Linear Models.
    This class implements Generalised Linear Models using the
    UCB, Thompson Sampling algorithms.
	Parameters
    ----------
	arms	: array, shape (n_arms, n_features) Set of arms for the UCB
			  algorithm.
    algo	: string, Algorithm used to solve the MAB problem. 'UCB' and
			  'Thompson Sampling' are the options
    solver 	: string, default: "logistic regression"
			  Model to estimate the best w_hat to be used for arm prediction
	ro		: float, exploration parameter to give weight to exploration in UCB
	nu		: float, covariance parameter to give weight to covariance in TS
    Attributes
    ----------
    w_hat	: array, shape (1, n_features) The leaning parameter for the
			  algorithm.
    """


	def __init__(self, arms, algo = 'lazy_UCB', ro = 0.5, nu = 0.5,
					solver = "logistic", warm_start = True):
		self.algo = algo
		self.arms = arms
		n_samples, n_features = self.arms.shape
		self.ro = ro
		self.nu = nu
		self.f = np.zeros(n_features,)
		self.w_hat = np.zeros(n_features,)
		self.M_ = np.eye(n_features)
		self.M_inv_ = np.eye(n_features)
		if (solver == "logistic"):
			self.model_ = LM.LogisticRegression(solver = 'newton-cg', C = 0.5)
			# self.model_ = LM.LogisticRegression(solver = 'newton-cg', \
												# warm_start = warm_start)
			self.link_ = "logistic"
		elif (solver == "linear"):
			self.model_ = LM.Ridge()
			self.link_ = "identity"
		else:
			assert False, "THIS MODEL NOT DEFINED"


	def fit(self, X, Y):
		"""Fit the model according to the given training data.
        Parameters
        ----------
        X : array, shape (n_samples, n_features) Training vector, where
			n_samples is the number of samples and n_features is the
			number of features.
        y : array-like, shape (n_samples, 1)
            Target vector relative to X.
        Returns
        -------
        self : object, Returns MAB estimator.
        """
		self.model_.fit(X, Y)
		self.w_hat = self.model_.coef_
		return self


	# def update_matrix(self, X, mu):
	def update_matrix(self, X):
		"""Update the self design matrix after new samples
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        -------
        self.M_inv_ : array, shape(n_features, n_features)
            Returns the design matrix.
        """
		# mu = helper.link_func(self.link_, np.dot(np.squeeze(X), np.squeeze(self.w_hat)))
		# self.M_ = self.M_ + mu * (1 - mu) * np.outer(X, X.transpose())
		# print X.shape
		n_samples, n_features = X.shape
		mu = []
		for i in range(n_samples):
			mu.append(helper.link_func(self.link_, np.dot(X[i], np.squeeze(self.w_hat))))
		# print mu
		mu = np.asmatrix(mu)
		mu2 = np.asmatrix(np.ones(n_samples)) - np.asmatrix(mu)
		mu = mu * X
		# print mu.shape
		mu2 = mu2 * X
		# print mu2.shape
		# print mu.T.shape
		sm = np.asarray(mu.T * mu2)
		# print sm.shape, sm[0]
		for i in range(n_features):
			# print self.M_[i][i], sm[i][i]
			self.M_[i][i] = self.M_[i][i] + sm[i][i]
		self.M_inv_ = np.linalg.inv(self.M_)
		# self.M_inv_ = helper.Lin_Alg.update_mat_inv(self.M_inv_, X, mu)


	def update(self, context, reward, choices = []):
		if self.link_ == "identity":
			self.update_matrix(context, reward)
			mu = helper.link_func(self.link_, np.dot(context, self.w_hat))
			self.f = self.f + reward * mu * (1 - mu) * context
			self.w_hat = np.dot(self.M_inv_, self.f.transpose())
		elif self.link_ == "logistic":
			l = len(choices)
			S = np.zeros((l, l))
			n_samples, n_features = self.arms.shape
			X = self.arms[choices[0]].reshape(1, n_features)
			mu = helper.link_func(self.link_, np.dot(self.arms[choices[0]], self.w_hat))
			S[0][0] = mu * (1 - mu)
			for i in range(1, l):
				X = np.concatenate((X, self.arms[choices[i]].reshape(1, n_features)), axis = 0)
				mu = helper.link_func(self.link_, np.dot(self.arms[choices[i]], self.w_hat))
				S[i][i] = mu * (1 - mu)
			X = np.asmatrix(X)
			S = np.asmatrix(S)
			self.M_ = X.T * S * X + np.asmatrix(np.eye(n_features))
			self.M_inv_ = np.linalg.inv(self.M_)
			self.w_hat = np.asarray(self.M_inv_ * X.T * S * np.asmatrix(reward).T)



	def predict_arm(self, acquisition_function):
		"""Function to predict next arm to be sampled
        Parameters
        ----------
		acquisition_function : callable, Function to balance exploitation
							   and exploration.
        Returns
        -------
        self.arm : array, shape(1, n_features)
            Returns the best arm to be pulled.
        """
		if self.algo == 'lazy_UCB':
			rewards = []
			# print "============================================="
			for arm in self.arms:
				rw = acquisition_function(arm)
				# print rw
				rewards.append(acquisition_function(arm))
			ch = np.argmax(np.asarray(rewards))
			# print ch
			# print self.w_hat
			return self.arms[ch]

		if self.algo == 'UCB':
			n_samples, n_features = self.arms.shape
			best_arm = self.arms[0]
			best_mu = helper.link_func(self.link_, np.dot(self.w_hat, self.arms[0]))
			# print self.w_hat.shape, self.w_hat[100]
			for i in range(1, n_samples):
				curr_mu = helper.link_func(self.link_, np.dot(self.w_hat, self.arms[i]))
				if curr_mu > best_mu:
					best_mu = curr_mu
					best_arm = self.arms[i]
			# print best_arm.shape
			self.update_matrix(best_arm.reshape(1, n_features))
			vec = np.dot(self.M_, best_arm.reshape(n_features,))
			# self.w_hat += (self.ro / np.linalg.norm(vec)) * vec
			self.w_hat += vec / np.linalg.norm(vec)
			self.w_hat /= np.linalg.norm(self.w_hat)
			return best_arm

		if self.algo == 'lazy_TS':
			# Get w_tilda from normal centered at w_hat
			try:
				n_samples, n_features = self.arms.shape
				w_tilde = np.random.multivariate_normal(np.squeeze(self.w_hat)\
							, self.nu * self.nu  * self.M_inv_, 1)[0]
				ch = np.argmax(np.dot(self.arms, w_tilde))
				return self.arms[ch], ch
			except:
				print self.M_inv_.shape
				sys.exit(0)

			# # Get arm which maximizes the dot product with the w_hat in case of finite arms
			# if n_features < float("inf"):
			# 	ip = np.dot(self.arms, w_tilde)
			# 	rewards = []
			# 	for i in ip:
			# 		rewards.append(helper.link_func(self.link_, i))
			# 	return self.arms[np.argmax(np.asarray(rewards))]
			# # Get arm as the unit vector along the w_hat in case of infinite arms
			# else:
			# 	return w_tilde / np.linalg.norm(w_tilde)

		assert False, "Algo not defined"



	def acquisition(self, arm):
		"""Acquisition function to balance exploitation and exploration
        Parameters
        ----------
		arm : array, shape(1, n_features) Arm for which to calculate the
			  score of acquisition function
        Returns
        -------
        exploitation + exploration : float,
            						Returns the acquisition score.
        """
		mu = helper.link_func(self.link_, np.dot(self.w_hat.transpose(), arm))
		# mu = np.dot(arm, np.squeeze(self.w_hat))
		explore = self.ro * np.sqrt(np.dot(np.dot(arm.transpose(), self.M_inv_), arm))

		return mu + explore




class Adversary:
	def __init__(self, w_star, X, model = "logistic"):
		self.w_star_ = w_star
		if (model == "logistic"):
			self.link_ = "logistic"
		elif (model == "linear"):
			self.link_ = "identity"
		else:
			assert False, "THIS MODEL NOT DEFINED"
		self.a_star_reward = np.amax(self.get_adversary_reward(X))

	def get_adversary_reward(self, X):
		n_samples, n_features = X.shape
		ip = np.dot(X, self.w_star_)
		rewards = []
		for i in ip:
			rewards.append(helper.link_func(self.link_, i))
		return np.asarray(rewards)
