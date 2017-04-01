import numpy as np
from sklearn.cluster import KMeans
import pickle
from os.path import exists

def get_data():
	pt = "../../dataset/covtype.data"
	pkl_dir = "../../pickle/"
	if exists(pkl_dir + "arms.p"):
		return pickle.load(open(pkl_dir + "arms.p")), \
				pickle.load(open(pkl_dir + "Y.p")), \
				pickle.load(open(pkl_dir + "lb.p"))
	X = []
	Y = [[] for i in range(32)]
	rw = []
	# arms = [np.zeros(10) for i in range(32)]
	with open(pt, "rb") as f:
		for line in f:
			ln = line.replace("\r","").replace("\n","").strip().split(",")
			ar = [float(ln[i]) for i in range(10)]
			ar.append(int(ln[-1]))
			# rw.append(int(ln[-1]))
			X.append(np.asarray(ar))
	l = len(X)
	X = np.asarray(X)
	tt = [0 for i in range(32)]
	lb = [0 for i in range(32)]
	print "data read"
	kmeans = KMeans(n_clusters=32, random_state=0).fit(X)
	print "clustering done"
	print kmeans.cluster_centers_.shape
	print type(kmeans.cluster_centers_)
	arms = kmeans.cluster_centers_
	for i in range(l):
		tt[kmeans.labels_[i]] += 1
		# arms[kmeans.labels_[i]] += X[i][:10]
		# arms[kmeans.labels_[i]] += X[i]
		# if rw[i] == 1:
		if X[i][-1] == 1:
			Y[kmeans.labels_[i]].append(1)
			lb[kmeans.labels_[i]] += 1
		else:
			Y[kmeans.labels_[i]].append(0)
	Y = np.asarray([np.asarray(i) for i in Y])
	for i in range(32):
		ft = float(tt[i])
		lb[i] = float(lb[i]) / ft
		# arms[i] = arms[i] / ft
	best = max(lb)
	print "best = ", best
	for i in range(32):
		lb[i] = best - lb[i]

	# lb, arms = (list(t) for t in zip(*sorted(zip(lb, arms))))
	# print sorted(lb)
	# print lb
	# arms = np.asarray(arms)
	pickle.dump(kmeans.cluster_centers_, open(pkl_dir + "arms.p", 'wb'))
	pickle.dump(Y, open(pkl_dir + "Y.p", 'wb'))
	pickle.dump(lb, open(pkl_dir + "lb.p", 'wb'))

	return kmeans.cluster_centers_, Y, lb