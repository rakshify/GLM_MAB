import numpy as np
from sklearn.cluster import KMeans
import pickle
from os.path import exists
from os import listdir
from os.path import isfile, join

pkl_dir = "../pickle/"

def get_data():
	pt = "../dataset/covtype.data"
	if exists(pkl_dir + "arms.p"):
		return pickle.load(open(pkl_dir + "arms.p")), \
				pickle.load(open(pkl_dir + "Y.p")), \
				pickle.load(open(pkl_dir + "lb.p"))

	# l = 0
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
	print X.shape
	tt = [0 for i in range(32)]
	lb = [0.0 for i in range(32)]
	# lb = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for i in range(32)]
	print "data read"
	if exists(pkl_dir + "kmeans.p"):
		kmeans = pickle.load(open(pkl_dir + "kmeans.p"))
	else:
		kmeans = KMeans(n_clusters=32, random_state=0).fit(X)
		pickle.dump(kmeans, open(pkl_dir + "kmeans.p", 'wb'))
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
			lb[kmeans.labels_[i]] += 1.0
		else:
			Y[kmeans.labels_[i]].append(0)

		# multinomial
		# Y[kmeans.labels_[i]].append(X[i][-1])
		# lb[kmeans.labels_[i]][int(X[i][-1] - 1)] += 1.0
	Y = np.asarray([np.asarray(i) for i in Y])
	for i in range(32):
		ft = float(tt[i])
		lb[i] = lb[i] / ft

		# multinomial
		# for j in range(7):
		# 	lb[i][j] = float(lb[i][j]) / ft
	# best = max(lb)
	# print "best = ", best
	# for i in range(32):
	# 	lb[i] = best - lb[i]

	# lb, arms = (list(t) for t in zip(*sorted(zip(lb, arms))))
	# print sorted(lb)
	# print lb
	# arms = np.asarray(arms)
	pickle.dump(kmeans.cluster_centers_, open(pkl_dir + "arms.p", 'wb'))
	pickle.dump(Y, open(pkl_dir + "Y.p", 'wb'))
	pickle.dump(lb, open(pkl_dir + "lb.p", 'wb'))

	return kmeans.cluster_centers_, Y, lb


def get_articles(pt):
	print "reading data"
	if exists(pkl_dir + "aid.p"):
		return pickle.load(open(pkl_dir + "aid.p"))
	aid = {}
	onlyfiles = [join(pt, f) for f in listdir(pt) if isfile(join(pt, f))]
	i = 0
	for file_path in onlyfiles:
		i += 1
		with open(file_path, "rb") as f:
			for line in f:
				s = [si[3:].strip() for si in line.replace("\n", "").replace("\r", "").split("|")[2:]]
				for si in s:
					if si not in aid:
						aid[si] = 1
		print "File%d read"%i

	aid = aid.keys()
	pickle.dump(aid, open(pkl_dir + "aid.p", 'wb'))

	return aid
