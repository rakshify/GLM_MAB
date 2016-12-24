from agent import ThompsonSampling, LinUCB, EpsGreedy, Random
from GLM_MAB import GLM_MAB, Adversary
import matplotlib.pyplot as plt
import numpy as np, pickle

my_features = 25

# Generate w_star for adversary
mean = np.zeros(my_features)
covariance = np.random.rand(my_features, my_features)
covariance = np.dot(covariance, covariance.transpose()) + np.eye(my_features)
# w_star = np.random.multivariate_normal(mean, covariance, 1)[0]
# # w_star = np.random.multivariate_normal(np.zeros(my_features), np.eye(my_features), 1)[0]
# w_star = w_star / np.linalg.norm(w_star)
#
# # Get arms
# X = np.random.multivariate_normal(mean, covariance, 100)
# # X = np.random.multivariate_normal(np.zeros(my_features), np.eye(my_features), 100)
# n_samples, n_features = X.shape
# for i in range(n_samples):
# 	X[i] = X[i] / np.linalg.norm(X[i])
# pickle.dump(X, open("./X.p", "wb"))
# pickle.dump(w_star, open("./ws.p", "wb"))
X = pickle.load(open("./X.p", "rb"))
w_star = pickle.load(open("./ws.p", "rb"))

a_star_reward = np.amax(np.dot(X, w_star))
# adversary = Adversary(w_star, X, model = "linear")

y_plot = []
# colors = ['red', 'green', 'blue', 'black']
# labels = []
colors = ['red', 'green']
labels = ["TS", "UCB"]
lines = [0, 0, 0, 0]
plt.subplot(211)
model = ThompsonSampling(nu = .1, d = my_features)
yp = []
regret = 0.0
avg_regret = 0.0
for t in range(1000):
    next_arm = X[model.choose(X)]
    reward = np.dot(next_arm, w_star)
    # reward = adversary.get_adversary_reward(next_arm.reshape(1, my_features))
    model.update(next_arm, reward)
    regret += a_star_reward - reward
    avg_regret = regret / (t + 1)
    yp.append(regret)
print avg_regret
lines[0], = plt.plot(range(1, 1001), yp, label = labels[0], color = colors[0])
y_plot.append([yp[ypi] / (ypi + 1) for ypi in range(len(yp))])
model = LinUCB(alpha = .5, d = my_features)
yp = []
regret = 0.0
avg_regret = 0.0
for t in range(1000):
    next_arm = X[model.choose(X)]
    reward = np.dot(next_arm, w_star)
    # reward = adversary.get_adversary_reward(next_arm.reshape(1, my_features))
    model.update(next_arm, reward)
    regret += a_star_reward - reward
    avg_regret = regret / (t + 1)
    yp.append(regret)
print avg_regret
lines[1], = plt.plot(range(1, 1001), yp, label = labels[1], color = colors[1])
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



# i = -1
# lines = [0, 0, 0, 0]
# plt.subplot(211)
# for nu in [1.0, .5, .7, .1]:
#     i += 1
#     model = ThompsonSampling(nu = nu, d = my_features)
#     yp = []
#     regret = 0.0
#     avg_regret = 0.0
#     for t in range(10000):
#         next_arm = X[model.choose(X)]
#         reward = np.dot(next_arm.transpose(), w_star)
#         model.update(next_arm, reward)
#         regret += a_star_reward - reward
#         avg_regret = regret / (t + 1)
#         yp.append(regret)
#     print avg_regret
#     labels.append("nu = " + str(nu))
#     lines[i], = plt.plot(range(1, 10001), yp, label = labels[i], color = colors[i])
#     y_plot.append([yp[ypi] / (ypi + 1) for ypi in range(len(yp))])
# plt.ylabel("Cumulative Regret")
# plt.xlabel("time steps")
# plt.subplot(212)
# i = -1
# for yp in y_plot:
#     i += 1
#     lines[i], = plt.plot(range(1, 10001), yp, label = labels[i], color = colors[i])
# plt.ylabel("Average Regret")
# plt.xlabel("time steps")
# plt.legend()
# plt.show()
