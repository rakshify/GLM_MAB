from GLM_MAB import GLM_MAB, Adversary
from math import log, sqrt
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import numpy as np, pylab, random





# sys.setrecursionlimit(1000000)

# Generate w_star for adversary
mean = np.zeros(1000)
covariance = np.random.rand(1000, 1000)
covariance = np.dot(covariance, covariance.transpose()) / np.linalg.norm(np.dot(covariance, covariance.transpose()))
D = 10
w_star = np.random.multivariate_normal(mean, covariance, size = 1)[0]
w_star = w_star / np.linalg.norm(w_star)

adversary = Adversary(w_star)

# Get arms
X = np.random.multivariate_normal(mean, covariance, size = 100)
for x in X:
	x = x / np.linalg.norm(x)



# Lists for plot
y_plots = []
x_plots = []
y_plot = []
x_plot = []
# bi = 0


colors = ['red', 'green', 'blue', 'brown']
labels = []
print "iter ", 1
const = .7
labels.append("const = " + str(const))
# print("Creating object")
ro = sqrt(const * log(50))
model = GLM_MAB(arms = X, w_hat = np.random.multivariate_normal(mean, covariance, size = 1)[0], ro = ro)
# print("object created")
# print(hasattr(model, 'get_params'))

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
# print("matrix updated")
model.fit(dataset["ip"], dataset["op"])
back_up = dataset
buy = Y
chosen = []
regret = np.linalg.norm(adversary.w_star_ - model.w_hat)
print regret
x_plot.append(50)
y_plot.append(regret)

for i in range(t, 1000):
	# if(i % 100 == 0):
	# 	print "Doing %d"%i

	ro = sqrt(const * log(i + 1))
	param = {'ro' : ro}
	model = model.set_params(param)
	# print model.solver

	# Predicting and sampling the next best arm
	next_arm = model.predict_arm(model.acquisition)					# Predit arm
	model.update_matrix(next_arm)					# Update design matrix
	chosen.append(np.where(model.arms == next_arm)[0][0])
	y = adversary.get_adversary_reward(next_arm)	# Sample and get reward from adversary

	Y1 = list(dataset["op"])
	X1 = list(dataset["ip"])
	X1.append(X[j])
	Y.append(y)
	Y1.append(bernoulli.rvs(y))
	dataset["ip"] = np.array(X1)
	dataset["op"] = np.array(Y1)

	model.fit(dataset["ip"], dataset["op"])
	regret = np.linalg.norm(adversary.w_star_ - model.w_hat)
	print regret
	x_plot.append(i + 1)
	y_plot.append(regret)
x_plots.append(x_plot)
# avg_reg = .001 * cum_regret
# print avg_reg
y_plots.append(y_plot)





for const in [1, 2, 5]:
    labels.append("const = " + str(const))
    y_plot = []
    x_plot = []
    dataset = back_up
    Y = buy
    print "iter ", const
    ro = sqrt(const * log(50))
    model = GLM_MAB(arms = X, w_hat = np.random.multivariate_normal(mean, covariance, size = 1)[0], ro = ro)
    t = 50

    model.update_matrix(dataset["ip"])
    model.fit(dataset["ip"], dataset["op"])
    regret = np.linalg.norm(adversary.w_star_ - model.w_hat)
    print regret
    x_plot.append(50)
    y_plot.append(regret)

    for i in range(t, 1000):
        ro = sqrt(const * log(i + 1))
        param = {'ro' : ro}
        model = model.set_params(param)

        # Predicting and sampling the next best arm
        next_arm = model.predict_arm(model.acquisition)					# Predit arm
        model.update_matrix(next_arm)					# Update design matrix
        chosen.append(np.where(model.arms == next_arm)[0][0])
        y = adversary.get_adversary_reward(next_arm)	# Sample and get reward from adversary

        Y1 = list(dataset["op"])
        X1 = list(dataset["ip"])
        X1.append(X[j])
        Y.append(y)
        Y1.append(bernoulli.rvs(y))
        dataset["ip"] = np.array(X1)
        dataset["op"] = np.array(Y1)

        model.fit(dataset["ip"], dataset["op"])
        regret = np.linalg.norm(adversary.w_star_ - model.w_hat)
        print regret
        x_plot.append(i + 1)
        y_plot.append(regret)
    x_plots.append(x_plot)
    y_plots.append(y_plot)

for i in range(len(x_plots)):
    plt.plot(x_plots[i], y_plots[i], colors[i], label = labels[i])


# plt.title('Average Regret curve at different values for const')
# pylab.xlim([0.1, 1])
plt.ylabel('Regret')
plt.xlabel('Time step')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.show()
