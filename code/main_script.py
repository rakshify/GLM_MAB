from GLM_MAB import GLM_MAB, Adversary
from learner import MAB_GridSearch as gs
from math import log, sqrt
import helper
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import bernoulli
import numpy as np, pylab, random, pickle
from time import time

my_features = 25

# Generate w_star for adversary
mean = np.zeros(my_features)
covariance = np.random.rand(my_features, my_features)
covariance = np.dot(covariance, covariance.transpose()) + np.eye(my_features)
w_star = np.random.multivariate_normal(mean, covariance, 1)[0]
# w_star = np.random.multivariate_normal(np.zeros(my_features), np.eye(my_features), 1)[0]
w_star = w_star / np.linalg.norm(w_star)

# Get arms
X = np.random.multivariate_normal(mean, covariance, 100)
# X = np.random.multivariate_normal(np.zeros(my_features), np.eye(my_features), 100)
n_samples, n_features = X.shape
for i in range(n_samples):
	X[i] = X[i] / np.linalg.norm(X[i])
# X = pickle.load(open("./X.p", "rb"))
# w_star = pickle.load(open("./ws.p", "rb"))

adversary = Adversary(w_star, X, model = "logistic")
# a_star_reward = np.amax(np.dot(X, w_star))

y_plot = []
# colors = ['red', 'green', 'blue', 'black']
# labels = []
colors = ['red', 'green']
labels = ["lazy_TS", "lazy_UCB"]
plt.subplot(211)
lines = [0, 0]
i = -1
for algo in labels:
	i += 1
	model = GLM_MAB(X, algo = algo, solver = "logistic")
	yp = []
	regret = 0.0
	avg_regret = 0.0
	for t in range(1000):
		next_arm = model.predict_arm(model.acquisition)
		next_arm = next_arm.reshape(1, n_features)
		reward = adversary.get_adversary_reward(next_arm)
		# reward = np.dot(next_arm, w_star)
		model.update(next_arm, reward[0])
		regret += adversary.a_star_reward - reward[0]
		avg_regret = regret / (t + 1)
		yp.append(regret)
	print avg_regret
	lines[i], = plt.plot(range(1, 1001), yp, label = labels[i], color = colors[i])
	y_plot.append([yp[ypi] / (ypi + 1) for ypi in range(len(yp))])
plt.ylabel("Cumulative Regret")
plt.xlabel("time steps")
plt.subplot(212)
i = -1
for yp in y_plot:
	i += 1
	lines[i], = plt.plot(range(1, 1001), yp, label = labels[i], color = colors[i])
plt.ylabel("Average Regret")
plt.xlabel("time steps")
plt.legend()
plt.show()


# plt.subplot(211)
# i = -1
# lines = [0, 0, 0, 0]
# for nu in [1.0, .5, .7, .1]:
# 	i += 1
# 	model = GLM_MAB(X, algo = "lazy_UCB", ro = nu, solver = "logistic")
# 	yp = []
# 	regret = 0.0
# 	avg_regret = 0.0
# 	for t in range(1000):
# 		next_arm = model.predict_arm(model.acquisition)
# 		next_arm = next_arm.reshape(1, n_features)
# 		reward = adversary.get_adversary_reward(next_arm)
# 		model.update(next_arm, reward)
# 		regret += adversary.a_star_reward - reward[0]
# 		avg_regret = regret / (t + 1)
# 		yp.append(regret)
# 	print avg_regret
# 	labels.append("ro = " + str(nu))
# 	lines[i], = plt.plot(range(1, 1001), yp, label = labels[i], color = colors[i])
# 	y_plot.append([yp[ypi] / (ypi + 1) for ypi in range(len(yp))])
# plt.ylabel("Cumulative Regret")
# plt.xlabel("time steps")
#
# plt.subplot(212)
# i = -1
# for yp in y_plot:
# 	i += 1
# 	lines[i], = plt.plot(range(1, 1001), yp, label = labels[i], color = colors[i])
# plt.ylabel("Average Regret")
# plt.xlabel("time steps")
# plt.legend()
# plt.show()
