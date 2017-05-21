import numpy as np, inspect, helper, sys, random, scipy
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


	def __init__(self, arms, algo = 'lazy_UCB', ro = 0.5, nu = 0.5, eps = 0.5,
					link = "logistic", warm_start = True, solver = "laplace"):
		self.algo = algo
		print arms.shape
		n_samples, n_features = arms.shape
		self.ro = ro
		self.nu = nu
		self.f = np.zeros(n_features,)
		self.sumf = np.zeros(n_features,)
		self.w_hat = np.zeros(n_features,)
		self.M_ = np.eye(n_features)
		self.M_inv_ = np.eye(n_features)
		self.link_ = link
		self.solver = solver
		self.sumeps = eps
		self.pulls = [0 for i in range(32)]
		if solver == "filippi":
			print "yes"
			self.rewards = []
			self.contexts = []
		if algo == "sgld":
			self.solver = "sgld"
		if (link == "logistic"):
			self.model_ = LM.LogisticRegression(solver = 'newton-cg', C = 0.5)
		elif (link == "linear"):
			self.model_ = LM.Ridge()
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
		if self.solver == "lse" or self.solver == "filippi":
			self.M_ = self.M_ + np.outer(X, X.transpose())
			# self.M_inv_ = helper.Lin_Alg.update_mat_inv(self.M_, X)
			self.M_inv_ = np.linalg.inv(self.M_)
		elif self.solver == "laplace":
			for x in X:
				x = np.squeeze(x)
				mu = helper.link_func(self.link_, np.dot(x, np.squeeze(self.w_hat)))
				for i in range(x.shape[0]):
					self.M_[i][i] = self.M_[i][i] + x[i] * x[i] * mu * (1 - mu)
			# self.M_inv_ = helper.Lin_Alg.update_mat_inv(self.M_, X)
			self.M_inv_ = np.linalg.inv(self.M_)


	def update(self, contexts, context, reward, alpha = .75, t = 0):
		# self.update_matrix(context)
		# print self.link_
		if self.solver == "lse":
			self.update_matrix(context)
			self.f = self.f + reward * context
			self.w_hat = np.dot(self.M_inv_, self.f.transpose())
		elif self.solver == "laplace":
			# self.w_hat = helper.gd(self.w_hat, self.M_, context, reward, alpha)
			self.w_hat = scipy.optimize.minimize(fun = self.__to_optimize_laplace, x0 = self.w_hat,\
													args = (self.M_, self.w_hat, context, reward)).x
			self.update_matrix(context)
		elif self.solver == "sgld":
			self.w_hat, self.f, self.M_ = helper.sgld(self.w_hat, self.f,\
											self.M_, context, reward, alpha)
		elif self.solver == "polya":
			# print context
			# print "===================="
			l = len(context)
			context = np.asarray(context)
			# print context
			reward = np.asarray(reward)
			# print context.shape
			# print self.w_hat.shape
			# print reward.shape, reward
			# print reward[-1]
			psi = np.dot(context, self.w_hat)
			# print t, alpha
			gamma = t ** (-alpha)
			kappa = reward - np.ones(l) * 5.0
			Omega = np.eye(l)
			# print context
			# print self.w_hat
			# print psi
			w = np.ones(l)
			if not 0 in psi:
				w = 5.0 * np.tanh(psi / 2.0) / psi
			for idx in range(l):
				Omega[idx][idx] = w[idx]
			
			# print Omega.shape
			# print context.shape
			# print self.M_.shape
			# print "##################"
			# print "##################"
			inter = np.asarray(np.asmatrix(context).T * np.asmatrix(Omega) * np.asmatrix(context))
			# print gamma
			# print inter
			# print inter.shape
			# print kappa.shape
			self.M_ = (1 - gamma) * self.M_ + gamma * inter
			# print kappa
			self.f = (1 - gamma) * self.f + gamma * np.dot(context.T, kappa)
			# print "gamma = ", gamma
			# print "inter = ", np.linalg.det(gamma * np.asmatrix(inter))
			# print np.linalg.det(np.asmatrix(self.M_))
			# print self.M_
			# print "##################"
			self.M_inv_ = np.linalg.inv(self.M_)
			# print self.M_inv_
			# print np.linalg.det(np.asmatrix(self.M_inv_))
			self.w_hat = np.dot(self.M_inv_, self.f)
		elif self.solver == "filippi":
			prev = self.w_hat
			if type(reward) == list:
				self.update_matrix(context[-1])
				self.rewards = reward
				self.contexts = context
				# print "list", self.rewards
				# l = len(reward)
				# for i in range(l):
				# 	self.rewards.append(reward[i])
				# 	self.contexts.append(context[i])
			else:
				# print "not list", reward
				self.update_matrix(context)
				self.rewards.append(reward)
				self.contexts.append(context)
			self.w_hat = scipy.optimize.root(self.__to_optimize, prev).x
		# if self.link_ == "identity":
		# 	self.update_matrix(context)
		# 	self.f = self.f + reward * context
		# 	self.w_hat = np.dot(self.M_inv_, self.f.transpose())
		# elif self.link_ == "logistic":
		# 	l = len(choices)
		# 	S = np.zeros((l, l))
		# 	n_samples, n_features = contexts.shape
		# 	X = contexts[choices[0]].reshape(1, n_features)
		# 	mu = helper.link_func(self.link_, np.dot(contexts[choices[0]], self.w_hat))
		# 	S[0][0] = mu * (1 - mu)
		# 	for i in range(1, l):
		# 		X = np.concatenate((X, contexts[choices[i]].reshape(1, n_features)), axis = 0)
		# 		mu = helper.link_func(self.link_, np.dot(contexts[choices[i]], self.w_hat))
		# 		S[i][i] = mu * (1 - mu)
		# 	X = np.asmatrix(X)
		# 	S = np.asmatrix(S)
		# 	self.M_ = X.T * S * X + np.asmatrix(np.eye(n_features))
		# 	self.M_inv_ = np.linalg.inv(self.M_)
		# 	self.w_hat = np.asarray(self.M_inv_ * X.T * S * np.asmatrix(reward).T)


	def __to_optimize(self, theta):
		to_sum = []
		# print len(self.rewards)
		for t in range(len(self.rewards)):
			to_sum.append(self.rewards[t] - helper.link_func(self.link_, np.inner(self.contexts[t], theta))\
							 * self.contexts[t])

		return np.sum(to_sum, 0)

	def __to_optimize_laplace(self, theta, Q, m, contexts, rewards):
		# print theta.shape
		# print Q.shape
		prior = 0.5 * np.dot((theta - m).T, np.dot(Q, (theta - m)))
		neg_likeli = 0
		l = len(contexts)
		for i in range(l):
			v = np.dot(theta, contexts[i])
			if(rewards[i] == 0):
				v *= -1
			neg_likeli += log(helper.logistic(v))

		return prior - neg_likeli



	def predict_arm(self, contexts, acquisition_function, eps = 0.0):
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
			for arm in contexts:
				rw = acquisition_function(arm)
				rewards.append(acquisition_function(arm))

			ch = np.argmax(np.asarray(rewards))
			self.pulls[ch] += 1
			return ch

		if self.algo == 'UCB':
			n_samples, n_features = contexts.shape
			best_arm = contexts[0]
			best_mu = helper.link_func(self.link_, np.dot(self.w_hat, contexts[0]))
			for i in range(1, n_samples):
				curr_mu = helper.link_func(self.link_, np.dot(self.w_hat, contexts[i]))
				if curr_mu > best_mu:
					best_mu = curr_mu
					best_arm = contexts[i]
			self.update_matrix(best_arm.reshape(1, n_features))
			vec = np.dot(self.M_, best_arm.reshape(n_features,))
			self.w_hat += vec / np.linalg.norm(vec)
			self.w_hat /= np.linalg.norm(self.w_hat)
			return best_arm

		if self.algo == 'lazy_TS':
			try:
				n_samples, n_features = contexts.shape
				w_tilde = np.random.multivariate_normal(np.squeeze(self.w_hat)\
							, self.nu * self.nu  * self.M_inv_, 1)[0]
				ch = np.argmax(np.dot(contexts, w_tilde))
				self.pulls[ch] += 1
				return ch#, w_tilde
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

		# if self.algo == "polya":

		if self.algo == "sgld":
			self.sumf += eta * self.f
			self.sumeps += eps
			self.w_hat = self.sumf / self.sumeps
			return np.argmax(np.dot(contexts, self.f))


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
		explore = self.ro * np.sqrt(np.dot(np.dot(arm.transpose(), self.M_inv_), arm))

		return mu + explore




class Adversary:
	def __init__(self, w_star, X, model = "logistic", log_bias = None):
		self.w_star_ = w_star
		if (model == "logistic"):
			self.link_ = "logistic"
		elif (model == "linear"):
			self.link_ = "identity"
		elif (model == "forest cover"):
			self.link_ = model
			self.log_bias_ = log_bias
			self.a_star_reward = np.amax(self.log_bias_)
			return
		else:
			assert False, "THIS MODEL NOT DEFINED"
		self.a_star_reward = np.amax(self.get_adversary_reward(X))

	def best_reward(self, contexts):
		n_samples, n_features = contexts.shape
		ip = np.dot(contexts, self.w_star_)
		rewards = []
		for i in ip:
			rewards.append(helper.link_func(self.link_, i))
		return np.amax(np.asarray(rewards))

	def get_adversary_reward(self, X):
		if self.link_ == "forest cover":
			l = len(self.w_star_[X])
			idx = random.randint(0, l - 1)
			# reg = self.a_star_reward - self.log_bias_[X]
			reg = self.log_bias_[X]
			return self.w_star_[X][idx], reg
		n_samples, n_features = X.shape
		ip = np.dot(X, self.w_star_)
		rewards = []
		for i in ip:
			rewards.append(helper.link_func(self.link_, i))
		return np.asarray(rewards)
