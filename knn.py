from collections import Counter 
from heapq import *

import math
import random

random.seed(42)

def euclid(x1, x2):
    return math.sqrt(sum(pow(a-b, 2) for a, b in zip(x1, x2)))

def get_train_test(X, y, frac=0.8, seed=42):
    inds = random.sample(range(len(X)), len(X))
    train = int(frac * len(X)) 
    return (X[:train], y[:train], X[train:], y[train:])

def accuracy(y_true, preds):
    return sum(a==b for a, b in zip(y_true, preds))/len(y_true)

class KNN:
    def __init__(self, k):
        self.k = k 

    def fit_predict(self, X_train, y_train, X_test):
        preds = []

        for sample in X_test:
            pq = []
            for i, row in enumerate(X_train):
                dist = euclid(sample, row)
                if len(pq) == self.k and -pq[0][0] > dist:
                    heappushpop(pq, (-dist, i))
                elif len(pq) < self.k:
                    heappush(pq, (-dist, i))
            c = Counter([y_train[i] for k, i in pq])
            # print(c.most_common(1))
            preds.append(c.most_common(1)[0][0])
        return preds 

def main():
    from sklearn import datasets 
    from sklearn.neighbors import KNeighborsClassifier

    data = datasets.load_iris()
    X, y = data.data, data.target

    trainx, trainy, testx, testy = get_train_test(X, y)

    knn = KNN(5)
    preds = knn.fit_predict(trainx, trainy, testx)
    print(accuracy(testy, preds))

    # baseline
    skl = KNeighborsClassifier(n_neighbors=5, p=2, metric="minkowski")
    preds1 = skl.fit(trainx, trainy).predict(testx)
    print(accuracy(testy, preds1))

main()
