import numpy as np, time
from math import sqrt, log
import helper


class Estimator:
	def __init__(self, dataset):
		self.dataset = dataset

	def estimate(self, name, start, D):
		if (name == "Gradient Descent"):
			return self.SGD(start, D)
		else:
			assert False, "This algorithm not supported!"


	def SGD(self, w_hat, D):
		iters = 0
		size = len(self.dataset["ip"])
		# print "size = ", size
		g = np.zeros_like(self.dataset["ip"][0])
		# print "dims = ", g.size
		for i in range(size):
			val = np.dot(w_hat, self.dataset["ip"][i])
			# print "val = ", val
			link = helper.logistic(val)
			mul = self.dataset["op"][i] - link
			# print mul, link
			g += mul * self.dataset["ip"][i]
		# g = sum([(self.dataset["op"][i] - helper.logistic(np.dot(w_hat, self.dataset["ip"][i]))) * self.dataset["ip"][i] ])
		# eta = .000001 / sqrt(iters)
		# nw_hat = w_hat + [eta * gi for gi in g]
		# norm = np.linalg.norm(nw_hat - w_hat)
		norm_g = np.linalg.norm(g)
		# g2 = [abs(gi) for gi in g]
		# w_hat = nw_hat
		# print norm_g
		# g = [gi / norm_g for gi in g]
		# g2 = [abs(gi) for gi in g]
		# while (np.max(g2) >= 0.00001):
		while (norm_g >= 0.00001):
			iters += 1
			# eta = D / (norm_g * sqrt(iters))
			eta = 1 / log(1 + iters)
			# print np.linalg.norm(w_hat)
			# print g
			# print type(eta), type(g[0])
			# nw_hat = w_hat + [eta * gi for gi in g]
			w_hat = w_hat + eta * g
			# print (eta * g).size
			# print np.linalg.norm(w_hat)
			# g = sum([(self.dataset["op"][i] - helper.logistic(np.dot(w_hat, self.dataset["ip"][i]))) * self.dataset["ip"][i] for i in range(size)])
			g = np.zeros_like(self.dataset["ip"][0])
			for i in range(size):
				mul = (self.dataset["op"][i] - helper.logistic(np.dot(w_hat, self.dataset["ip"][i])))
				g += mul * self.dataset["ip"][i]
			# norm = np.linalg.norm(nw_hat - w_hat)
			norm_g = np.linalg.norm(g)
			# w_hat = nw_hat
			# g2 = [abs(gi) for gi in g]
			# print norm_g
			# if(iters % 1000 == 0):
			# 	print "here ", norm_g
			# g = [gi / norm_g for gi in g]
			# g2 = [abs(gi) for gi in g]
			# print norm_g
			# print "======================"
			# time.sleep(1)

		return w_hat
