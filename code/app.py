#!flask/bin/python
from flask import Flask, jsonify
import numpy as np, sys
sys.path.insert(0, "/home/rakshit/Desktop/Academic/ms_codefundo/News_Recommender/policy/")
sys.path.insert(0, "/home/rakshit/Desktop/Academic/ms_codefundo/News_Recommender/environment/")
import utilities as utils
from glmUCB import GLMUCB
from linUCB import LinUCB
from epsilon_greedy import EpsilonGreedy
from oblivious import Oblivious
from adversary import Adversary
import matplotlib.pyplot as plt, requests, random, unicodedata, json


app = Flask(__name__)
aid = []

@app.route('/todo/api/v1.0/get_next_article', methods=['GET'])
def get_tasks():
    l = len(aid)
    while(True):
        idx = random.randint(0, l - 1)
        url = "https://webhose.io/search?token=f2543ae7-df09-4c15-8004-f425eed71bcb&format=json&q="+aid[idx]
        r  = requests.get(url).text
        data = json.loads(unicodedata.normalize('NFKD', r).encode('ascii','ignore'))
        if(len(data["posts"]) > 0):
            break
    # print data["posts"]
    # print url
    result = {'article_id': aid[idx], 'article': data}
    return jsonify(result)

if __name__ == '__main__':
    X, Y, lb = utils.get_data()
    pt = "../dataset/ms"
    aid = utils.get_articles(pt)
    print "articles read"
    print aid
    adv = Adversary(Y, X, model = "forest cover", log_bias = lb)
    print "DATA READY"

    y_plot = []
    # colors = ['red', 'green', 'blue', 'black']
    colors = ['red', 'green', 'blue', 'black', 'brown', 'pink', 'orange', 'violet', 'cyan', 'yellow']
    labels = []
    plt.subplot(211)
    lines = [0 for i in range(1, 11)]
    i = -1
    for algo in ["glmUCB"]:
        i += 1
        if algo == "glmUCB":
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

        for t in range(1000):
            contexts = X
            ch = model.predict_arm(contexts)
            reward, reg = adv.get_adversary_reward(ch)
            model.pull_arm(ch, reward, contexts)

            regret += reg
            if t % 100 == 0:
                print regret, t/100
            avg_regret = regret / (t + 1)
            yp.append(regret)

        print avg_regret
        # print dist_diff[-1]
        labels.append("algo = " + str(algo))
        lines[i], = plt.plot(range(1, 1001), yp, label = labels[i], color = colors[i])
        y_plot.append([yp[ypi] / (ypi + 1) for ypi in range(len(yp))])
        # y_plot.append(dist_diff)
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
    app.run(host = "172.27.20.230")