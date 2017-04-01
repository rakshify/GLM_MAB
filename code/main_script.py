from GLM_MAB import GLM_MAB, Adversary
from learner import MAB_GridSearch as gs
from math import log, sqrt
import helper
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import bernoulli
import numpy as np, pylab, random, pickle
from time import time
import utilities as utils

# my_features = 25

# # Generate w_star for adversary
# mean = np.zeros(my_features)
# covariance = np.random.rand(my_features, my_features)
# covariance = np.dot(covariance, covariance.transpose()) + np.eye(my_features)
# w_star = np.random.multivariate_normal(mean, covariance, 1)[0]
# # w_star = np.random.multivariate_normal(np.zeros(my_features), np.eye(my_features), 1)[0]
# w_star = w_star / np.linalg.norm(w_star)

# # Get arms
# X = np.random.multivariate_normal(mean, covariance, 100)
# # X = np.random.multivariate_normal(np.zeros(my_features), np.eye(my_features), 100)
# n_samples, n_features = X.shape
# for i in range(n_samples):
# 	X[i] = X[i] / np.linalg.norm(X[i])
# # X = pickle.load(open("./X.p", "rb"))
# # w_star = pickle.load(open("./ws.p", "rb"))

# # adversary = Adversary(w_star, X, model = "linear")
# adversary = Adversary(w_star, X)


X, Y, lb = utils.get_data()
adversary = Adversary(Y, X, model = "forest cover", log_bias = lb)
print "DATA READY"

y_plot = []
# colors = ['red', 'green', 'blue', 'black']
colors = ['red', 'green', 'blue', 'black', 'brown', 'pink', 'orange', 'violet', 'cyan', 'yellow']
labels = []
plt.subplot(211)
lines = [0 for i in range(1, 11)]

# utils.filippi(X, adversary)

i = -1
# for nu in [1]:
# for ro in range(1, 2):
# for alpha in [65]:
# for alpha in range(55, 100, 10):
for ro in range(1, 11, 10):
	i += 1
	# alpha = float(alpha) / 100.0
	ro = float(ro) / 1000.0
	model = GLM_MAB(X, algo = "lazy_UCB", ro = ro, solver = "polya")
	# model = GLM_MAB(X, algo = "lazy_UCB", ro = ro, solver = "filippi")
	yp = []
	dist_diff = []
	regret = 0.0
	avg_regret = 0.0
	# dataset = {}
	# dataset['ip'] = []
	# dataset['op'] = []
	# # h = np.zeros(100)
	# for t in range(10):
	# 	idx = np.unique(np.random.randint(100, size=20))
	# 	contexts = X[idx,:]
	# 	next_arm = contexts[int(np.random.uniform(1, len(contexts), 1)[0]) - 1]
	#
	# 	reward = adversary.get_adversary_reward(next_arm.reshape(1, n_features))
	# 	choices.append(list(next_arm))
	# 	dataset['ip'].append(np.squeeze(next_arm))
	# 	dataset['op'].append(bernoulli.rvs(reward, size = 1)[0])
	# 	regret += adversary.best_reward(contexts) - reward[0]
	# 	avg_regret = regret / (t + 1)
	# 	yp.append(regret)
	#
	# model.fit(dataset['ip'], dataset['op'])
	# model.update_matrix(np.asmatrix(choices))
	choices = []
	ys = []

	# reward = None
	# for t in range(10, 1000):
	for t in range(100000):
		# idx = np.unique(np.random.randint(100, size=20))
		# contexts = X[idx,:]
		contexts = X
		# model.ro = sqrt(3 * log(max(2, t + 1)))
		ch = model.predict_arm(contexts, model.acquisition)
		next_arm = contexts[ch]
		choices.append(next_arm)
		# next_arm = next_arm.reshape(1, n_features)
		# reward = adversary.get_adversary_reward(next_arm)
		reward, reg = adversary.get_adversary_reward(ch)
		# log_reward = bernoulli.rvs(reward[0], size = 1)[0]
		# print reward
		# ys.append(log_reward)
		ys.append(reward)
		# print ys
		# choices.append(list(np.squeeze(next_arm)))
		# if t == 0:
		# 	reward = adversary.get_adversary_reward(next_arm)
		# else:
		# 	rt = adversary.get_adversary_reward(next_arm)
		# 	reward = np.concatenate((reward, rt), axis = 0)
		# dataset['ip'].append(np.squeeze(next_arm))
		# dataset['op'].append(bernoulli.rvs(reward, size = 1)[0])
		# model.fit(dataset['ip'], dataset['op'])
		# model.update(contexts, choices, ys, alpha = alpha)
		# choices = []
		# ys = []
		if t != 0:
			model.update(contexts, choices, ys, alpha = .65, t = t + 1)
			# model.update(contexts, choices, ys, t = t + 1)
		# model.update(contexts, choices, ys)
		# model.update(contexts, choices, ys, alpha = alpha, t = t + 1)

		# regret += adversary.best_reward(contexts) - reward[0]
		regret += reg
		# if (t + 1) % 10 == 0:
		# 	model.update(contexts, choices, ys, alpha = .65, t = t + 1)
		# 	# model.update(contexts, choices, ys, t = t + 1)
		# 	choices = []
		# 	ys = []
		if t % 1000 == 0:
			print regret, t/1000
			# print lb
			lbb = []
			for b in lb:
				lbb.append(b)
			lbb, pulls = (list(t) for t in zip(*sorted(zip(lbb, model.pulls))))
			# print lb
			# print lbb
			print pulls
		avg_regret = regret / (t + 1)
		yp.append(regret)
		# dt = 0.0
		# smp = 1
		# wt = np.random.multivariate_normal(np.squeeze(model.w_hat)\
		# 			, model.nu * model.nu  * model.M_inv_, smp)
		# for j in range(smp):
		# 	dtj = abs(np.dot(adversary.w_star_, wt[j]))
		# 	dtj /= np.linalg.norm(wt[j]) * np.linalg.norm(adversary.w_star_)
		# 	dt += dtj
		# dt /= smp
		# dist_diff.append(dt)
		# dt = abs(np.dot(adversary.w_star_, model.w_hat))
		# dt /= np.linalg.norm(model.w_hat) * np.linalg.norm(adversary.w_star_)
		# yp.append(dt)
		ys = []
		choices = []
	print avg_regret
	# print dist_diff[-1]
	labels.append("ro = " + str(ro))
	lines[i], = plt.plot(range(1, 100001), yp, label = labels[i], color = colors[i])
	y_plot.append([yp[ypi] / (ypi + 1) for ypi in range(len(yp))])
	# y_plot.append(dist_diff)
plt.ylabel("Cumulative regret")
plt.xlabel("time steps")

plt.subplot(212)
i = -1
for yp in y_plot:
	i += 1
	lines[i], = plt.plot(range(1, 100001), yp, label = labels[i], color = colors[i])
plt.ylabel("Average regret")
plt.xlabel("time steps")
plt.legend()
plt.show()
