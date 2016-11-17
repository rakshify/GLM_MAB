import numpy as np, time, itertools, pickle
from math import sqrt, log
import helper
from scipy.stats import bernoulli
from time import time

#
# class Estimator:
# 	def __init__(self, dataset):
# 		self.dataset = dataset
#
# 	def estimate(self, name, start, D):
# 		if (name == "Gradient Descent"):
# 			return self.SGD(start, D)
# 		else:
# 			assert False, "This algorithm not supported!"
#
#
# 	def SGD(self, w_hat, D):
# 		iters = 0
# 		size = len(self.dataset["ip"])
# 		# print "size = ", size
# 		g = np.zeros_like(self.dataset["ip"][0])
# 		# print "dims = ", g.size
# 		for i in range(size):
# 			val = np.dot(w_hat, self.dataset["ip"][i])
# 			# print "val = ", val
# 			link = helper.logistic(val)
# 			mul = self.dataset["op"][i] - link
# 			# print mul, link
# 			g += mul * self.dataset["ip"][i]
# 		# g = sum([(self.dataset["op"][i] - helper.logistic(np.dot(w_hat, self.dataset["ip"][i]))) * self.dataset["ip"][i] ])
# 		# eta = .000001 / sqrt(iters)
# 		# nw_hat = w_hat + [eta * gi for gi in g]
# 		# norm = np.linalg.norm(nw_hat - w_hat)
# 		norm_g = np.linalg.norm(g)
# 		# g2 = [abs(gi) for gi in g]
# 		# w_hat = nw_hat
# 		# print norm_g
# 		# g = [gi / norm_g for gi in g]
# 		# g2 = [abs(gi) for gi in g]
# 		# while (np.max(g2) >= 0.00001):
# 		while (norm_g >= 0.00001):
# 			iters += 1
# 			# eta = D / (norm_g * sqrt(iters))
# 			eta = 1 / log(1 + iters)
# 			# print np.linalg.norm(w_hat)
# 			# print g
# 			# print type(eta), type(g[0])
# 			# nw_hat = w_hat + [eta * gi for gi in g]
# 			w_hat = w_hat + eta * g
# 			# print (eta * g).size
# 			# print np.linalg.norm(w_hat)
# 			# g = sum([(self.dataset["op"][i] - helper.logistic(np.dot(w_hat, self.dataset["ip"][i]))) * self.dataset["ip"][i] for i in range(size)])
# 			g = np.zeros_like(self.dataset["ip"][0])
# 			for i in range(size):
# 				mul = (self.dataset["op"][i] - helper.logistic(np.dot(w_hat, self.dataset["ip"][i])))
# 				g += mul * self.dataset["ip"][i]
# 			# norm = np.linalg.norm(nw_hat - w_hat)
# 			norm_g = np.linalg.norm(g)
# 			# w_hat = nw_hat
# 			# g2 = [abs(gi) for gi in g]
# 			# print norm_g
# 			# if(iters % 1000 == 0):
# 			# 	print "here ", norm_g
# 			# g = [gi / norm_g for gi in g]
# 			# g2 = [abs(gi) for gi in g]
# 			# print norm_g
# 			# print "======================"
# 			# time.sleep(1)
#
# 		return w_hat





class MAB_GridSearch:
	"""Grid Search for some Multi Arm Bandit.
	This class is the model to tune hyperparameters for the
	estimator using grid search algorithm.
	Uses k-fold cross validation for scoring.
	Parameters
	----------
	estimator	: object, The estimator object. This is the black box
				  machine learning model.
	adversary	: object, Adversary for this estimator for scoring
	params		: dict, key = Parameter name in estimator
				  value = Space of values for the parameter to search into.
				  Parmaters set to exhaustively search upon
	samples		: int, Sampling complexity for the estimator arms
	k		  	: Parameter for cross validation.
	plot		: object of matplotlib.pyplot, Plot object if needed to plot
				  intermediate results
	Attributes
	----------
	best_estimator_ : object, The estimator object equipped with best
					 set of parameters.
	"""


	def __init__(self, estimator, adversary, params, samples,
					k = 5, plot = None):
		self.estimator = estimator
		self.adversary = adversary
		self.params = params
		# self.score = score
		self.samples = samples
		self.k = k
		self.plot = plot
		self.sp = 1


	def fit(self, X, Y):
		"""Fit the model according to the given training data.
		Parameters
		----------
		X : array, shape (n_samples, n_features) Training vector, where
			n_samples is the number of samples and n_features is the
			number of features.
		Y : array-like, shape (n_samples, 1)
			Target vector relative to X.
		Returns
		----------
		self : object, Returns grid search model.
		"""
		n_samples, n_features = X.shape
		# Number of partitions for cross validation
		l = (n_samples + self.k - 1) / self.k
		config_dict = {}

		# Make list of sets of parameters and their values space
		param_names = []
		param_lists = []
		for param in self.params:
			param_names.append(param)
			param_lists.append(self.params[param])

		# Some useful variables
		config = 0
		self.best_params = {}
		best_score = 0.0
		x_plot = []
		y_plot = []

		# Loop over all configurations to check for best
		for param_list in itertools.product(*param_lists):
			cur_estimator = self.estimator
			Des_mat = np.zeros((n_features, n_features))
			for x in cur_estimator.arms:
				Des_mat = Des_mat + np.outer(x, x.transpose())
			config += 1
			config_dict[config] = param_list
			regret = 0.0
			const = 0
			# Make dictionary of parameters for this configuration
			params = {}
			num_params = len(param_names)
			for j in range(num_params):
				params[param_names[j]] = param_list[j]
			if 'ro' in params:
				const = params['ro']
				params['ro'] = sqrt(const * log(np.linalg.norm(Des_mat)))
			if 'nu' in params:
				const = params['nu']
				params['nu'] = const * log(50)

			# Set this configuration in the estimator
			cur_estimator.set_params(params)

			# pt = time()

			cur_estimator.fit(X, Y)											#Initial FIT
			# print "Took %f seconds in initial fit"%(time() - pt)
			diff = self.adversary.w_star_ - cur_estimator.w_hat
			regret = np.linalg.norm(diff)
			print "regret = " + str(regret) + "\twhile taking 0th step"

			# Sample arms further
			for t in range(self.samples):
				# pt = time()
				next_arm = cur_estimator.predict_arm(cur_estimator.acquisition)		# Predit arm
				# print "Took %f seconds to predict arm"%(time() - pt)
				# pt = time()
				# print next_arm.shape
				cur_estimator.update_matrix(next_arm)						# Update design matrix
				y = self.adversary.get_adversary_reward(next_arm)			# Sample and get reward from adversary

				# Add the predicted arm and reward in list
				Y1 = list(Y)
				X1 = list(X)
				X1.append(next_arm)
				Y1.append(bernoulli.rvs(y))
				X = np.array(X1)
				Y = np.array(Y1)

				# Fit again and update the difference
				cur_estimator.fit(X, Y)
				diff = self.adversary.w_star_ - cur_estimator.w_hat
				regret += np.linalg.norm(diff)
				print "regret = " + str(regret) + "\twhile taking " + str(t) + "th step"

				# Update design matrix and set ro for further samples
				Des_mat = Des_mat + np.outer(next_arm, next_arm.transpose())
				if 'ro' in params:
					params['ro'] = sqrt(const * log(np.linalg.norm(Des_mat)))
				if 'nu' in params:
					params['nu'] = const * log(51 + t)
				cur_estimator.set_params(params)
				# print "Took %f seconds to update for this arm"%(time() - pt)
			# Average regret for this config
			avg_regret = float(regret) / float(self.samples + n_samples)
			print "average regret = " + str(avg_regret) + "\twhile taking " + str(config) + "th configuration"
			x_plot.append(config)
			y_plot.append(avg_regret)
			# If first configuration or a better configuration,
			# update the best configuration
			if config == 1 or avg_regret < best_score:
				best_score = avg_regret
				self.best_params = params

		# Plot graphs before leaving
		if self.plot is not None:
			self.plot.subplot(2, 1, 1)
			if self.estimator.algo == 'UCB':
				self.plot.plot(x_plot, y_plot, 'red')
			else:
				self.plot.plot(x_plot, y_plot, 'blue')
			self.plot.ylabel('average regret')
			self.plot.xlabel('configuration')

		# Update the best estimator with the best parameter configuration
		self.best_estimator_ = self.estimator.set_params(self.best_params).fit(X, Y)
		if self.estimator.algo == 'UCB':
			pickle.dump(config_dict, open("./configUCB.p", "wb"))
		else:
			pickle.dump(config_dict, open("./configTS.p", "wb"))
		self.sp += 2

		return self
