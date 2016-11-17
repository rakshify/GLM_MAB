from GLM_MAB import GLM_MAB, Adversary
from learner import MAB_GridSearch as gs
from math import log, sqrt
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import numpy as np, pylab, random
from time import time



# Generate w_star for adversary
mean = np.zeros(1000)
covariance = np.random.rand(1000, 1000)
covariance = np.dot(covariance, covariance.transpose()) / np.linalg.norm(np.dot(covariance, covariance.transpose()))
D = 10
pt = time()
w_star = np.random.multivariate_normal(mean, covariance, size = 1)[0]
ct = time()
# print "Took %f seconds to sample w_star"%(ct - pt)
# pt = ct
w_star = w_star / np.linalg.norm(w_star)

adversary = Adversary(w_star)

# Get arms
X = np.random.multivariate_normal(mean, covariance, size = 100)
n_samples, n_features = X.shape
for x in X:
	x = x / np.linalg.norm(x)

model = GLM_MAB(arms = X)

# Initialize grid search
params = {'ro' : [float(i + 1) / 100.0 for i in range(10, 1001, 10)]}
# ct = time()
# print "Took %f seconds in preprocessing"%(ct - pt)
# pt = ct
gs_model = gs(model, adversary, params, 100, plot = plt)

# Sample some points for warm start
dataset = {}
dataset["ip"] = []
dataset["op"] = []
Y = []
t = 50
h = np.zeros(1000)
for i in range(t):
	j = random.randint(0, 99)
	while(h[j] == 1):
		j = random.randint(0, 99)
	h[j] = 1
	dataset["ip"].append(model.arms[j])
	y = adversary.get_adversary_reward(model.arms[j])
	Y.append(y)
	dataset["op"].append(bernoulli.rvs(y))
dataset["ip"] = np.array(dataset["ip"])
dataset["op"] = np.array(dataset["op"])
model.update_matrix(dataset["ip"])

# Fit the model to get best estimator
model = gs_model.fit(dataset["ip"], dataset["op"]).best_estimator_
print "ro parameter for UCB = ", model.ro
print "Best const config = ", gs_model.best_params

# ct = time()
# print "Took %f seconds in sampling 50 points"%(ct - pt)

# Train on this best model
model.fit(dataset["ip"], dataset["op"])

y_plot = []
x_plot = []
regret = np.linalg.norm(adversary.w_star_ - model.w_hat)
print regret
x_plot.append(50)
y_plot.append(regret)

for i in range(t, 10000):
	next_arm = model.predict_arm(model.acquisition)					# Predit arm
	model.update_matrix(next_arm)					# Update design matrix
	# chosen.append(np.where(model.arms == next_arm)[0][0])
	y = adversary.get_adversary_reward(next_arm)	# Sample and get reward from adversary

	Y1 = list(dataset["op"])
	X1 = list(dataset["ip"])
	X1.append(next_arm)
	Y.append(y)
	Y1.append(bernoulli.rvs(y))
	dataset["ip"] = np.array(X1)
	dataset["op"] = np.array(Y1)

	model.fit(dataset["ip"], dataset["op"])
	regret = np.linalg.norm(adversary.w_star_ - model.w_hat)
	print "regret = " + str(regret) + "\twhile taking " + str(i) + "th step"
	x_plot.append(i + 1)
	y_plot.append(regret)

plt.subplot(2, 1, 2)
plt.plot(x_plot, y_plot, 'red')
plt.ylabel('Regret')
plt.xlabel('Time step')


model = GLM_MAB(arms = X, algo = 'TS')

# Initialize grid search
params = {'nu' : [float(i + 1) / 100.0 for i in range(10, 1001, 10)]}
gs_model = gs(model, adversary, params, 100, plot = plt)

# Sample some points for warm start
dataset = {}
dataset["ip"] = []
dataset["op"] = []
Y = []
t = 50
h = np.zeros(1000)
for i in range(t):
	j = random.randint(0, 99)
	while(h[j] == 1):
		j = random.randint(0, 99)
	h[j] = 1
	dataset["ip"].append(model.arms[j])
	y = adversary.get_adversary_reward(model.arms[j])
	Y.append(y)
	dataset["op"].append(bernoulli.rvs(y))
dataset["ip"] = np.array(dataset["ip"])
dataset["op"] = np.array(dataset["op"])
model.update_matrix(dataset["ip"])

# Fit the model to get best estimator
model = gs_model.fit(dataset["ip"], dataset["op"]).best_estimator_
print "nu parameter for TS = ", model.nu
print "Best const config = ", gs_model.best_params

# Train on this best model
model.fit(dataset["ip"], dataset["op"])

y_plot = []
x_plot = []
regret = np.linalg.norm(adversary.w_star_ - model.w_hat)
print regret
x_plot.append(50)
y_plot.append(regret)

for i in range(t, 10000):
	next_arm = model.predict_arm(model.acquisition)					# Predit arm
	model.update_matrix(next_arm)					# Update design matrix
	# chosen.append(np.where(model.arms == next_arm)[0][0])
	y = adversary.get_adversary_reward(next_arm)	# Sample and get reward from adversary

	Y1 = list(dataset["op"])
	X1 = list(dataset["ip"])
	X1.append(next_arm)
	Y.append(y)
	Y1.append(bernoulli.rvs(y))
	dataset["ip"] = np.array(X1)
	dataset["op"] = np.array(Y1)

	model.fit(dataset["ip"], dataset["op"])
	regret = np.linalg.norm(adversary.w_star_ - model.w_hat)
	print "regret = " + str(regret) + "\twhile taking " + str(i) + "th step"
	x_plot.append(i + 1)
	y_plot.append(regret)

plt.subplot(2, 1, 2)
plt.plot(x_plot, y_plot, 'blue')
plt.ylabel('Regret')
plt.xlabel('Time step')
plt.show()
