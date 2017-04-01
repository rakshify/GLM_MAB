from GLM_MAB import GLM_MAB, Adversary
from learner import MAB_GridSearch as gs
from math import log, sqrt
import helper
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import bernoulli
import numpy as np, pylab, random, pickle
from time import time
from sklearn import linear_model as LM

my_dict = {"job" : {"admin" : 42}, "mar" : {"divorced" : 42},\
            "edu" : {"basic.4y" : 42}, "def" : {"no" : 42}, "hou" : {"no" : 42},\
            "loan" : {"no" : 42}, "pout" : {"failure" : 42}}
ind = 1
for job in ["blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"]:
    my_dict["job"][job] = ind
    ind += 1
# print ind
for mar in ["married", "single", "unknown"]:
    my_dict["mar"][mar] = ind
    ind += 1
# print ind
for edu in ["basic.6y", "basic.9y", "high.school", "illiterate", "professional.course", "university.degree", "unknown"]:
    my_dict["edu"][edu] = ind
    ind += 1
# print ind
for df in ["yes", "unknown"]:
    my_dict["def"][df] = ind
    ind += 1
# print ind
for hou in ["yes", "unknown"]:
    my_dict["hou"][hou] = ind
    ind += 1
# print ind
for loan in ["yes", "unknown"]:
    my_dict["loan"][loan] = ind
    ind += 1
# print ind
ind = 32
for pout in ["nonexistent", "success"]:
    my_dict["pout"][pout] = ind
    ind += 1
# print ind


def fill_in(xi, line, field):
    if my_dict[field][line] < 40:
        xi[my_dict[field][line]] = 1


def fill(Y, line):
    xi = np.zeros(39)
    s = [i.strip().strip("'").strip('"').strip(".") for i in line.split(";")]
    xi[0] = float(s[0])
    fill_in(xi, s[1], "job")
    fill_in(xi, s[2], "mar")
    fill_in(xi, s[3], "edu")
    fill_in(xi, s[4], "def")
    fill_in(xi, s[5], "hou")
    fill_in(xi, s[6], "loan")
    for i in range(10, 14):
        xi[i + 18] = float(s[i])
    fill_in(xi, s[14], "pout")
    for i in range(15, 20):
        xi[i + 19] = float(s[i])
    if s[-1] == "yes":
        Y.append(1)
    if s[-1] == "no":
        Y.append(0)

    return xi.reshape(1, 39)


def get_coeff_vec(fpath):
    X = None
    Y = []
    with open(fpath, "r+") as f:
        fline = f.readline()
        X = fill(Y, f.readline())
        for line in f:
            X = np.concatenate((X, fill(Y, line)), axis = 0)
    Y = np.asarray(Y)

    return X, Y

if __name__ == "__main__":
    fp = "../../dataset/bank-additional/bank-additional.csv"
    tX, tY = get_coeff_vec(fp)
    fp = "../../dataset/bank-additional/bank-additional-full.csv"
    X, Y = get_coeff_vec(fp)
    mdl = LM.LogisticRegression(class_weight = "balanced")
    n_samples, n_features = X.shape
    print n_samples / (2 * np.bincount(Y))
    mdl.fit(X, Y)
    pY = mdl.predict(tX)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(tY)):
        if tY[i] == 1:
            if tY[i] == pY[i]:
                tp += 1
            else:
                fn += 1
        else:
            if tY[i] == pY[i]:
                tn += 1
            else:
                fp += 1
    print "tp\ttn\tfp\tfn"
    print str(tp) + "\t" + str(tn) + "\t" + str(fp) + "\t" + str(fn)
    w_star = np.squeeze(mdl.coef_)
    adversary = Adversary(w_star, X)
    y_plot = []
    colors = ['red', 'green', 'blue', 'black', 'brown', 'pink', 'orange', 'violet', 'cyan', 'yellow']
    labels = []
    plt.subplot(211)
    lines = [0 for i in range(1, 11)]
    i = -1
    for nu in [2, 3, 5]:
        i += 1
    	nu = float(nu) / 10.0
        model = GLM_MAB(X, algo = "lazy_TS", nu = nu)
    	yp = []
    	dist_diff = []
    	regret = 0.0
    	avg_regret = 0.0

        for t in range(1000):
            choices = []
            ys = []
            idx = np.unique(np.random.randint(n_samples - 1, size=20))
            contexts = X[idx,:]
            ch = model.predict_arm(contexts, model.acquisition)
            next_arm = contexts[ch]
            choices.append(next_arm)
            next_arm = next_arm.reshape(1, n_features)
            reward = adversary.get_adversary_reward(next_arm)
            ys.append(bernoulli.rvs(reward, size = 1)[0])
            model.update(contexts, choices, ys)
            regret += adversary.best_reward(contexts) - reward[0]
            if t % 100 == 0:
                print regret, t / 100
                # print w_star
                # print model.w_hat
            avg_regret = regret / (t + 1)
            yp.append(regret)
            # dt = 0.0
            # smp = 1
            # wt = np.random.multivariate_normal(np.squeeze(model.w_hat)\
    		# 			, model.nu * model.nu  * model.M_inv_, smp)
            # for j in range(smp):
            #     v1 = wt[j] / np.linalg.norm(wt[j])
            #     v2 = adversary.w_star_ / np.linalg.norm(adversary.w_star_)
            #     dtj = abs(np.dot(v1, v2))
            #     dt += dtj
            # dt /= smp
            # dist_diff.append(dt)
        print avg_regret
        # print dist_diff[-1]
        labels.append("nu = " + str(nu))
        print "i = ", i
        lines[i], = plt.plot(range(1, 1001), yp, label = labels[i], color = colors[i])
        # y_plot.append(dist_diff)
        y_plot.append([yp[ypi] / (ypi + 1) for ypi in range(len(yp))])

    plt.ylabel("Cumulative regret")
    plt.xlabel("time steps")

    plt.subplot(212)
    i = -1
    for yp in y_plot:
        i += 1
        lines[i], = plt.plot(range(1, 1001), yp, label = labels[i], color = colors[i])
    plt.ylabel("Average regret")
    plt.xlabel("time steps")
    plt.legend()
    plt.show()
