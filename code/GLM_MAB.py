import helper
from learner import Estimator
import numpy as np, random
from math import fabs
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from sklearn import linear_model as LM
from sklearn.model_selection import GridSearchCV


class GLM_MAB:
	"""Multi Arm Bandit Problem in Generalised Linear Models.
    This class implements Generalised Linear Models using the
    UCB, Thompson Sampling algorithms.
	Parameters
    ----------
    algo_	: string, Algorithm used to solve the MAB problem. 'UCB' and
			  'Thompson Sampling' are the options
    arms_	: array, shape (n_arms, n_features) Set of arms for the UCB
			  algorithm.
    model_ 	: bool, default: sklearn.linear_model.LogisticRegression
			  Model to estimate the best w_hat to be used for arm prediction
	ro		: float, exploration parameter to give weight to exploration
    Attributes
    ----------
    w_hat_	: array, shape (1, n_features) The leaning parameter for the
			  algorithm.
    """


	def __init__(self, algo = 'UCB', arms, w_hat, ro = 0.1
					model = "logistic regression"):
		self.algo_ = algo
		self.arms_ = arms
		self.w_hat_ = w_hat
		self.run = 0
		self.ro_ = ro
		if (model == "logistic regression")
			self.model_ = LM.LogisticRegression()
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
        self.model_ : object
            Returns theta estimator.
        """
		self.model_.fit(X, Y)
		self.w_hat_ = self.model_.coef_



	def update_matrix(self, X):
		"""Update the self design matrix after new samples
        Parameters
        ----------
        X : array, shape (1, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        -------
        self.M_inv_ : array, shape(n_features, n_features)
            Returns the design matrix.
        """
		if (self.run == 0):
			self.M_inv_ = np.zeros((len(X), len(X)))
			for x in X:
				self.M_inv_ = self.M_inv_ + np.outer(x, x.transpose())
			self.M_inv_ = np.linalg.pinv(self.M_inv_)
		else:
			add = np.outer(X, X.transpose())
			self.M_inv_ = helper.lin_alg.update_mat_inv(M_inv, add)
		self.run += 1



	def goodness_score(self, w_star):
		"""Scoring function to be used to given to Grid Search
        Parameters
        ----------
        w_star : array, shape (1, n_features)
            Parameter used by adversary to set the distribution over arms
        Returns
        -------
        epsilon : float
            Returns the l2 distance between actual and estimated parameter.
        """
		return np.linalg.norm(w_star - self.w_hat_)



	def predict_arm(self, acquisition_function = self.acquisition):
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
		best_arm = self.arms_[0]
		best_arm_score = acquisition(best_arm)
		for i in range(1, len(self.arms_)):
			arm_score = acquisition(self.arms_[i])
			if(best_arm_score < arm_score):
				best_arm_score = arm_score
				best_arm = self.arms_[i]
		return best_arm



	def acquisition(arm):
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
		mu = helper.link_func(self.link_, arm, self.w_hat_)
		explore = self.ro_ * np.dot(np.dot(arm, self.M_inv_), arm.transpose())

		return mu + explore




class Adversary:
	def __init__(self, w_star, model = "logistic regression"):
		self.w_star_ = w_star
		if (model == "logistic regression")
			self.link_ = "logistic"
		else:
			assert False, "THIS MODEL NOT DEFINED"

	def get_adversary_reward(*argv):
		argv.append(self.w_star_)
		return helper.link_func(self.link_, *argv)




# Generate w_star for adversary
mean = np.zeros(1000)
covariance = np.random.rand(1000, 1000)
covariance = np.dot(covariance, covariance.transpose()) / np.linalg.norm(np.dot(covariance, covariance.transpose()))
D = 10
w_star = np.random.multivariate_normal(mean, covariance, size = 1)[0]
w_star = w_star / np.linalg.norm(w_star)

adversary = Adversary(w_star)

# Get arms
X = np.random.multivariate_normal(mean, covariance, size = 1000)
for x in X:
	x = x / np.linalg.norm(x)

model = GLM_MAB(arms = X, w_hat = np.random.multivariate_normal(mean, covariance, size = 1)[0]))


# Lists for plot
y_plot = []
x_plot = []

dataset = {}
dataset["ip"] = []
dataset["op"] = []
Y = []
t = 50

for i in range(t):
	j = random.randint(0, 999)
	# M = M + np.outer(X[j], X[j].transpose())
	dataset["ip"].append(model.arms_[j])
	y = adversary.get_adversary_reward(model.arms_[j])
	Y.append(y)
	dataset["op"].append(bernoulli.rvs(y))

dataset["ip"] = np.array(dataset["ip"])
dataset["op"] = np.array(dataset["op"])
model.update_matrix(dataset["ip"])


# TO DO FIT THE MODEL USING GRID SEARCH

chosen = []

for i in range(t, 10001):
	print "Doing %d"%i

	# CALCULATION TO BE DONE
	ro = get_ro_value()

	# Predicting and sampling the next best arm
	next_arm = model.predict_arm()					# Predit arm
	model.update_matrix(next_arm)					# Update design matrix
	chosen.append(np.where(model.arms_ == next_arm)[0][0])
	y = adversary.get_adversary_reward(next_arm)	# Sample and get reward from adversary
	# Update the dataset with this sample
	Y1 = list(dataset["op"])
	X1 = list(dataset["ip"])
	X1.append(X[j])
	Y.append(y)
	Y1.append(bernoulli.rvs(y))
	dataset["ip"] = np.array(X1)
	dataset["op"] = np.array(Y1)


	# TO DO FIT THE MODEL USING GRID SEARCH

	x_plot.append(i)

plt.plot(x_plot, y_plot, 'ro')
plt.show()
