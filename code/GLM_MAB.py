# from learner import Estimator
import numpy as np, inspect, helper
from math import fabs, log, sqrt
from sklearn import linear_model as LM
from time import time
# from sklearn.base import BaseEstimator
# from sklearn.model_selection import GridSearchCV


# class GLM_MAB(BaseEstimator):
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


	def __init__(self, arms, algo = 'UCB', ro = 0.1, nu = 0.1,
					solver = "logistic regression", warm_start = True):
		self.algo = algo
		self.arms = arms
		# self.w_hat = w_hat
		self.run = 0
		self.ro = ro
		self.nu = nu
		self.params_list_ = dict()
		# print(solver)
		self.solver = solver
		frame = inspect.currentframe()
		args, _, _, values = inspect.getargvalues(frame)
		for i in args:
			self.params_list_[i] = values[i]
		self.params_list_.pop('self', None)
		if (solver == "logistic regression"):
			self.model_ = LM.LogisticRegression(solver = 'newton-cg', warm_start = warm_start)
			# self.params_list_['solver'] = self.model_
			self.link_ = "logistic"
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


	def predict(self, X):
		"""This function predicts the reward that we get from the estimators w_hat
        Parameters
        ----------
        X : array, shape (n_samples, n_features) Training vector, where
			n_samples is the number of samples and n_features is the
			number of features.
        Returns
        -------
        Y : float
            Returns rewards on current estimator.
        """
		n_samples, n_features = X.shape
		# print(n_samples, n_features)
		# print(X[0].shape)
		# print(self.w_hat.shape)
		# print(np.dot(self.w_hat, X[0]))
		Y = [helper.link_func(self.link_, np.dot(self.w_hat, x)[0]) for x in X]
		Y = np.array(Y)

		return Y


	def get_params(self, deep=True):
		"""Get parameters for this estimator.
		Parameters
		----------
		deep : boolean, optional
			If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
		"""
		# return self.params_list_
		out = dict()
		for key in self.params_list_:
			# We need deprecation warnings to always be on in order to
			# catch deprecated param values.
			# This is set in utils/__init__.py but it gets overwritten
			# when running under python3 somehow.
			# warnings.simplefilter("always", DeprecationWarning)
			# try:
				# with warnings.catch_warnings(record=True) as w:
					# value = getattr(self, key, None)
				# if len(w) and w[0].category == DeprecationWarning:
				# 	# if the parameter is deprecated, don't show it
				# 	continue
			# finally:
			# 	warnings.filters.pop(0)
			value = getattr(self, key, None)

			# XXX: should we rather test if instance of estimator?
			if deep and hasattr(value, 'get_params'):
				deep_items = value.get_params().items()
				out.update((key + '__' + k, val) for k, val in deep_items)
			out[key] = value
		return out


	def set_params(self, parameters):
		# print(type(parameters))
		# print(parameters)
		for param in parameters:
			# print "old = ", getattr(self, param)
			if(parameters[param] != 0):
				setattr(self, param, parameters[param])
			# print "new = ", getattr(self, param)

		return self



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
		if (self.run == 0):
			n_samples, n_features = X.shape
			self.M_inv_ = np.zeros((n_features, n_features))
			for x in X:
				self.M_inv_ = self.M_inv_ + np.outer(x, x.transpose())
			self.M_ = self.M_inv_
			self.M_inv_ = np.linalg.pinv(self.M_inv_)
		else:
			# add = np.outer(X, X.transpose())
			self.M_ = self.M_ + np.outer(X, X.transpose())
			self.M_inv_ = helper.Lin_Alg.update_mat_inv(self.M_inv_, X)
		self.run += 1



	def score(self, X, Y):
		"""Scoring function to be used to given to Grid Search
        Parameters
        ----------
        X : array, shape (n_samples, n_features) Training vector, where
			n_samples is the number of samples and n_features is the
			number of features.
        y : array-like, shape (n_samples, 1)
            Target vector relative to X.
        -------
        goodness : float
            Returns the difference of predicted rewards and actual rewards.
        """
		n_samples, n_features = X.shape
		Y_pred = self.predict(X)
		goodness = 0
		for i in range(n_samples):
			goodness += fabs(Y_pred[i] - Y[i])

		return goodness


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
		if self.algo == 'UCB':
			best_arm = self.arms[0]
			pt = time()
			best_arm_score = acquisition_function(best_arm)
			# print "Took %f seconds to calculate aquisiton function for 1 arm"%(time() - pt)
			for i in range(1, len(self.arms)):
				arm_score = acquisition_function(self.arms[i])
				if(best_arm_score < arm_score):
					best_arm_score = arm_score
					best_arm = self.arms[i]
			# print "Took %f seconds to get best arm"%(time() - pt)
			return best_arm
		elif self.algo == 'TS':
			# Get w_tilda from normal centered at w_hat
			n_samples, n_features = self.arms.shape
			pt = time()
			# w_tilde = np.random.multivariate_normal(self.w_hat.reshape((n_features,)), self.nu * self.M_, size = 1)[0]
			w_tilde = helper.gaussian(self.w_hat.reshape((n_features,)), self.nu * self.M_, 1)[0]
			# print "Took %f seconds to sample w_tilde"%(time() - pt)

			# Get arm which maximizes the dot product with the w_hat in case of finite arms
			if n_features < float("inf"):
				mx = -1
				flag = True
				best_arm = None
				pt = time()
				for arm in self.arms:
					if flag or np.dot(arm, w_tilde) > mx:
						mx = np.dot(arm, w_tilde)
						best_arm = arm
				# print "Took %f seconds to get best arm"%(time() - pt)
				return best_arm
			# Get arm as the unit vector along the w_hat in case of infinite arms
			else:
				return w_tilde / np.linalg(w_tilde)
		else:
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
		mu = helper.link_func(self.link_, np.dot(self.w_hat, arm))
		explore = self.ro * np.dot(np.dot(arm, self.M_inv_), arm.transpose())

		return mu + explore




class Adversary:
	def __init__(self, w_star, model = "logistic regression"):
		self.w_star_ = w_star
		if (model == "logistic regression"):
			self.link_ = "logistic"
		else:
			assert False, "THIS MODEL NOT DEFINED"

	def get_adversary_reward(self, x):
		# tuple(list(argv).append(self.w_star_))
		return helper.link_func(self.link_, np.dot(self.w_star_, x))
