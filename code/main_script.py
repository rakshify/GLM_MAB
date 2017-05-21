import numpy as np, sys
# sys.path.insert(0, "/home/rakshit/Dropbox/thesis/News_Recommender/")
sys.path.insert(0, "./policies/")
sys.path.insert(0, "./environment/")
import utilities as utils
from glmUCB import GLMUCB
from GLM_MAB import GLM_MAB
from linUCB import LinUCB
from epsilon_greedy import EpsilonGreedy
from oblivious import Oblivious
from adversary import Adversary
import matplotlib.pyplot as plt

X, Y, lb = utils.get_data()
adv = Adversary(Y, X, model = "forest cover", log_bias = lb)
print "DATA READY"

y_plot = []
# colors = ['red', 'green', 'blue', 'black']
colors = ['red', 'green', 'blue', 'black', 'brown', 'pink', 'orange', 'violet', 'cyan', 'yellow']
labels = []
plt.subplot(211)
lines = [0 for i in range(1, 11)]
model = None
i = -1
for (algo, arg) in [("polya", "lazy_TS"), ("glmUCB", "")]:#, ("polya", "lazy_UCB")]:
	i += 1
	if algo == "polya":
		model = GLM_MAB(X, algo = arg, solver = "polya")
	elif algo == "glmUCB":
		model = GLMUCB(X)
	elif algo == "oblivious":
		model = Oblivious(X)
	elif algo == "linUCB":
		model = LinUCB(X)
	else:
		model = EpsilonGreedy(X)
	yp = []
	dist_diff = []
	regret = 0.0
	avg_regret = 0.0
	choices = []
	ys = []

	for t in range(10000):
		contexts = X
		ch = None
		if algo == "polya":
			ch = model.predict_arm(contexts, model.acquisition)
			next_arm = contexts[ch]
			choices.append(next_arm)
		else:
			ch = model.predict_arm(contexts)
		reward, reg = adv.get_adversary_reward(ch)

		if algo == "polya":
			ys.append(reward)
			if t!= 0:
				model.update(contexts, choices, ys, alpha = .75, t = t + 1)
			# if t >= 1000:
			# 	choices[t % 1000] = choices[-1]
			# 	ys[t % 1000] = ys[-1]
			# 	choices.pop()
			# 	ys.pop()
			if t % 100 == 0:
				print regret, t/100
				# print lb
				lbb = []
				for b in lb:
					lbb.append(b)
				lbb, pulls = (list(t) for t in zip(*sorted(zip(lbb, model.pulls))))
				# print lb
				# print lbb
				print pulls
		else:
			model.pull_arm(ch, reward, contexts)

		regret += reg
		if t % 100 == 0:
			print regret, t/100
		avg_regret = regret / (t + 1)
		yp.append(regret)
		# choices = []
		# ys = []

	print avg_regret
	# print dist_diff[-1]
	labels.append("algo = " + str(algo))
	lines[i], = plt.plot(range(1, 10001), yp, label = labels[i], color = colors[i])
	y_plot.append([yp[ypi] / (ypi + 1) for ypi in range(len(yp))])
	# y_plot.append(dist_diff)
plt.ylabel("Cumulative regret")
plt.xlabel("time steps")

plt.subplot(212)
i = -1
for yp in y_plot:
	i += 1
	lines[i], = plt.plot(range(1, 10001), yp, label = labels[i], color = colors[i])
plt.ylabel("Average regret")
plt.xlabel("time steps")
plt.legend()
plt.show()
