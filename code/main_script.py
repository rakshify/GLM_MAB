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
# # a_star_reward = np.amax(np.dot(X, w_star))
#
# y_plot = []
# # colors = ['red', 'green', 'blue', 'black']
# # labels = []
# colors = ['red', 'green']
# labels = ["lazy_TS", "lazy_UCB"]
# plt.subplot(211)
# lines = [0, 0]
# i = -1
# for algo in labels:
# 	i += 1
# 	model = GLM_MAB(X, algo = algo, solver = "logistic")
# 	yp = []
# 	regret = 0.0
# 	avg_regret = 0.0
# 	for t in range(1000):
# 		next_arm = model.predict_arm(model.acquisition)
# 		next_arm = next_arm.reshape(1, n_features)
# 		reward = adversary.get_adversary_reward(next_arm)
# 		# reward = np.dot(next_arm, w_star)
# 		model.update(next_arm, reward[0])
# 		regret += adversary.a_star_reward - reward[0]
# 		avg_regret = regret / (t + 1)
# 		yp.append(regret)
# 	print avg_regret
# 	lines[i], = plt.plot(range(1, 1001), yp, label = labels[i], color = colors[i])
# 	y_plot.append([yp[ypi] / (ypi + 1) for ypi in range(len(yp))])
# plt.ylabel("Cumulative Regret")
# plt.xlabel("time steps")
# plt.subplot(212)
# i = -1
# for yp in y_plot:
# 	i += 1
# 	lines[i], = plt.plot(range(1, 1001), yp, label = labels[i], color = colors[i])
# plt.ylabel("Average Regret")
# plt.xlabel("time steps")
# plt.legend()
# plt.show()








y_plot = []
# colors = ['red', 'green', 'blue', 'black']
colors = ['red', 'green', 'blue', 'black', 'brown', 'pink', 'orange', 'violet', 'cyan', 'yellow']
labels = []
# colors = ['red', 'green']
# labels = ["linTS", "logit_TS"]
# plt.subplot(211)
# plt.title("smoothness factor = 0.2")
# lines = [0, 0]
# model1 = GLM_MAB(X, algo = "lazy_TS", solver = "logistic")
# model2 = GLM_MAB(X, algo = "lazy_TS", solver = "logistic")
# yp1 = []
# regret1 = 0.0
# avg_regret1 = 0.0
# yp2 = []
# regret2 = 0.0
# avg_regret2 = 0.0
# dataset = {}
# dataset['ip'] = []
# dataset['op'] = []
# h = np.zeros(100)
# for t in range(5):
# 	s = int(np.random.uniform(1,100,1)[0]) - 1
# 	while h[s] != 0:
# 		s = np.random.uniform(1,100,1)[0] - 1
# 	h[s] = 1
# 	model2.update_matrix(X[s])
# 	reward = adversary.get_adversary_reward(X[s].reshape(1, n_features))
# 	dataset['ip'].append(np.squeeze(X[s]))
# 	dataset['op'].append(bernoulli.rvs(reward, size = 1)[0])
# 	regret2 += adversary.a_star_reward - reward[0]
# 	avg_regret2 = regret2 / (t + 1)
# 	yp2.append(regret2)
# 	next_arm = model1.predict_arm(model1.acquisition)
# 	next_arm = next_arm.reshape(1, n_features)
# 	reward = adversary.get_adversary_reward(next_arm)
# 	# reward = np.dot(next_arm, w_star)
# 	model1.update(next_arm, reward[0])
# 	regret1 += adversary.a_star_reward - reward[0]
# 	avg_regret1 = regret1 / (t + 1)
# 	yp1.append(regret1)
# for t in range(5, 1000):
# 	next_arm = model2.predict_arm(model2.acquisition)
# 	next_arm = next_arm.reshape(1, n_features)
# 	reward = adversary.get_adversary_reward(next_arm)
# 	model2.update_matrix(next_arm)
# 	reward = adversary.get_adversary_reward(next_arm)
# 	dataset['ip'].append(np.squeeze(next_arm))
# 	dataset['op'].append(bernoulli.rvs(reward, size = 1)[0])
# 	regret2 += adversary.a_star_reward - reward[0]
# 	avg_regret2 = regret2 / (t + 1)
# 	yp2.append(regret2)
# 	next_arm = model1.predict_arm(model1.acquisition)
# 	next_arm = next_arm.reshape(1, n_features)
# 	reward = adversary.get_adversary_reward(next_arm)
# 	model1.update(next_arm, reward[0])
# 	regret1 += adversary.a_star_reward - reward[0]
# 	avg_regret1 = regret1 / (t + 1)
# 	yp1.append(regret1)
# print avg_regret1, avg_regret2
# lines[0], = plt.plot(range(1, 1001), yp1, label = labels[0], color = colors[0])
# lines[1], = plt.plot(range(1, 1001), yp2, label = labels[1], color = colors[1])
# y_plot.append([yp1[ypi] / (ypi + 1) for ypi in range(len(yp1))])
# y_plot.append([yp2[ypi] / (ypi + 1) for ypi in range(len(yp2))])
# plt.ylabel("Cumulative Regret")
# plt.xlabel("time steps")
# plt.subplot(212)
# i = -1
# for yp in y_plot:
# 	i += 1
# 	lines[i], = plt.plot(range(1, 1001), yp, label = labels[i], color = colors[i])
# plt.ylabel("Average Regret")
# plt.xlabel("time steps")
# plt.legend()
# plt.show()








plt.subplot(211)
lines = [0 for i in range(1, 11)]

# for nu in [5, 6, 8, 9, 10, 13, 14, 15, 16, 17]:
# for nu in [.5]:
i = -1
# for nu in range(1, 11):
for nu in [16]:
	i += 1
	nu = float(nu) / 10
	model = GLM_MAB(X, algo = "lazy_TS", nu = nu, solver = "logistic")
	yp = []
	regret = 0.0
	avg_regret = 0.0
	dataset = {}
	dataset['ip'] = []
	dataset['op'] = []
	h = np.zeros(100)
	choices = []
	for t in range(10):
		s = int(np.random.uniform(1,100,1)[0]) - 1
		while h[s] != 0:
			s = int(np.random.uniform(1,100,1)[0]) - 1
		h[s] = 1
		reward = adversary.get_adversary_reward(X[s].reshape(1, n_features))
		choices.append(list(X[s]))
		dataset['ip'].append(np.squeeze(X[s]))
		dataset['op'].append(bernoulli.rvs(reward, size = 1)[0])
		regret += adversary.a_star_reward - reward[0]
		avg_regret = regret / (t + 1)
		yp.append(regret)

	model.fit(dataset['ip'], dataset['op'])
	model.update_matrix(np.asmatrix(choices))

	# reward = None
	# for t in range(10, 1000):
	for t in range(10, 1000):
		# print t
		next_arm, ch = model.predict_arm(model.acquisition)
		next_arm = next_arm.reshape(1, n_features)
		reward = adversary.get_adversary_reward(next_arm)
		choices.append(list(np.squeeze(next_arm)))
		# if t == 0:
		# 	reward = adversary.get_adversary_reward(next_arm)
		# else:
		# 	rt = adversary.get_adversary_reward(next_arm)
		# 	reward = np.concatenate((reward, rt), axis = 0)
		dataset['ip'].append(np.squeeze(next_arm))
		dataset['op'].append(bernoulli.rvs(reward, size = 1)[0])
		model.fit(dataset['ip'], dataset['op'])
		# print np.asmatrix(choices)
		model.update_matrix(np.asmatrix(choices))
		# model.update(next_arm, reward, choices)
		# model.update(next_arm, reward)
		regret += adversary.a_star_reward - reward[0]
		if t % 100 == 0:
			print regret, t / 100
		avg_regret = regret / (t + 1)
		yp.append(regret)
	print avg_regret
	labels.append("nu = " + str(nu))
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
