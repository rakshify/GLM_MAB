import numpy as np, random


class Adversary:
	def __init__(self, w_star, X, model = "logistic", log_bias = None):
		self.w_star_ = w_star
		self.log_bias_ = log_bias
		self.a_star_reward = np.amax(self.log_bias_)
		self.best_arm = np.argmax(self.log_bias_)

		# multinomial
		# self.best_arm, self.a_star_reward = self.get_best_real_reward_multi()

	def get_adversary_reward(self, X):
		l = len(self.w_star_[X])
		l2 = len(self.w_star_[self.best_arm])
		idx = random.randint(0, l - 1)
		idx2 = random.randint(0, l2 - 1)
		rew = self.w_star_[X][idx]
		idx = np.unique(np.random.randint(100, size=1))
		# idx2 = np.unique(np.random.randint(100, size=10))
		# print self.w_star_[X], type(self.w_star_[X])
		# print self.w_star_[X][idx]
		# rew = np.sum(self.w_star_[X][idx])
		# print rew
		# reg = self.log_bias_[X]
		br = self.log_bias_[self.best_arm]
		rew = self.log_bias_[X]
		reg = br - rew

		# For multinomial
		# reg = self.a_star_reward - self.get_real_reward_multi(X)
		return rew, reg

	def get_best_real_reward_multi(self):
		best = 0
		br = 0.0
		for i in range(32):
			cr = self.get_real_reward_multi(i)
			if cr > br:
				best = i
				br = cr

		return best, br

	def get_real_reward_multi(self, i):
		cr = 0.0
		for j in range(7):
			cr += (j + 1) * self.log_bias_[i][j]

		return cr
