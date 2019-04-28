"""Predict missing multitask.

"""
import os 
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder


class Miss(object):
    def __init__(self):
        pass
    
    def load_data(self):
        months = set()
        years = set()
        y_train = []
        y_test = []
        X_train = []
        month_feat_train = []
        month_feat_test = []
        X_test = []

        N = int(input())
        headers = input().split()
        for i in range(N):
            line = input().split()
            years.add(line[0])
            months.add(line[1])
            temp = []
            
            if line[2][0] == 'M':
                y_test.append(line[2])
                temp.append(line[3])
                # temp.append(line[0][0])
                # temp.append(line[1][0])
                temp.append(1)
                # 1 if tmax missing (tmin is feature) and 0 if tmin(missing)
                X_test.append(temp)
                month_feat_test.append(line[1])
            elif line[3][0] == 'M':
                y_test.append(line[3])
                temp.append(float(line[2]))
                # temp.append(line[0][0])
                # # temp.append(line[1][0])
                temp.append(0)
                X_test.append(temp)
                month_feat_test.append(line[1])
            else:
                # train data; copy for tmin and tmax pred
                y_train.append(float(line[2]))
                temp.append(float(line[3]))
                # temp.append(line[0][0])
                # temp.append(line[1][0])
                temp.append(1)
                X_train.append(temp)
                month_feat_train.append(line[1])
                # and for label is tmin
                temp = []
                y_train.append(float(line[3]))
                temp.append(float(line[2]))
                # temp.append(line[0])
                # temp.append(line[1])
                temp.append(0)
                X_train.append(temp)
                month_feat_train.append(line[1])

        # forgo motnhs and years featurization for now
        # come back if time and need more performance

        # print(months)
        # print(years)
        # print(X_train)

        # month dict
        d = {m:i for m, i in zip(list(months), list(range(len(months))))}
        
        mt_train = self._preprocess_m(month_feat_train, d)
        mt_test = self._preprocess_m(month_feat_test, d)
        X_train = np.concatenate([np.array(X_train), mt_train.reshape(-1,1)], axis=1)
        X_test = np.concatenate([np.array(X_test), mt_test.reshape(-1,1)], axis=1)
        # print(mt_test.reshape(-1,1).shape)
        # print(np.array(X_test).shape)
        return X_train, np.array(y_train), X_test

    def _preprocess_m(self, mx, d):
        mt = []
        for m in mx:
            mt.append(d[m])
        return np.array(mt)

    def train(self, X_train, y_train):
        self.m = RandomForestRegressor(n_estimators=2000)
        self.m.fit(X_train, y_train)

    def predict(self, X_test):
        preds = self.m.predict(X_test)
        for i in range(len(preds)):
            print(preds[i])

    def __call__(self):
        X_train, y_train, X_test = self.load_data()
        self.train(X_train, y_train)
        self.predict(X_test)

def main():
    H = Miss()
    H()

if __name__ == '__main__':
    main()
