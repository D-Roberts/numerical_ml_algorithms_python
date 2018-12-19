#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Problem: Projected gradient descent so that half the weights are non-negative.
This is an alternative to lagrangian approach to constraints.

"""

import numpy as np


def gradient(beta, X, y):
    Xt = np.transpose(X)

    return (1.0 / X.shape[0]) * (2 * np.dot((np.dot(np.transpose(X), X)), beta) - 2 * np.dot(Xt, y))


def loss(preds, y):
    err = preds - y

    return (1.0 / preds.shape[0]) * np.dot(np.transpose(err), err)


def add_intercept(X):
    intercept = np.ones(X.shape[0], dtype=np.float32).reshape((X.shape[0], 1))

    return np.concatenate((X, intercept), axis=1)

def solution1(X, y):
    # Type your solution here
    # Run 10,000 steps of projected gradient descent

    epochs = 10000
    lr = 0.1
    X_train = add_intercept(X)

    # initialize betas

    example_w = np.abs(np.random.randn(20))
    example_b = np.random.randn()

    beta = np.append(example_w, example_b)

    # forward pass
    epoch = 1
    while True:
        grad = gradient(beta, X_train, y)
        beta_prev = beta
        beta = beta - lr * grad

        # project onto space
        for j in range(10):
            if beta[j] < 0:
                beta[j] = 0
        epoch += 1

        if epoch >= epochs:
            break

    return beta[:20], beta[20]


class LR_SGD(object):
    def __init__(self, X, y, epochs, lr, tol=1e-5):
        self.X = X
        self.y = y
        self.epochs = epochs
        # batch is all data
        self.lr = lr
        self.tol = tol

    def train(self):
        X_train = add_intercept(self.X)

        # initialize betas
        beta = np.array([0] * X_train.shape[1])

        for i in range(1, X_train.shape[1]):
            beta[i] = 0.5

        # forward pass
        epoch = 1

        while True:
            grad = gradient(beta, X_train, self.y)
            beta_prev = beta
            beta = beta - self.lr * grad
            # project onto space
            for j in range(10):
                if beta[j] < 0:
                    beta[j] = 0
            epoch += 1

            if epoch % 5:
                print('beta shape', beta.shape)
                print('X shape', X_train.shape)

                preds = self.predict(beta, X_train)

                # loss on train set
                print('Loss at epoch {} is {}'.format(epoch, loss(preds, self.y)))

            if all(beta - beta_prev < self.tol) or epoch >= self.epochs:
                break

        return beta

    def predict(self, beta, X_pred):

        # assumes intercept added

        return np.dot(X_pred, beta)


def main():
    X = np.random.random((100, 20))
    y = np.random.random(100)
    # univariate here
    # X = X.reshape((X.shape[0], 1))
    print(X.shape)
    print(y.shape)

    lr_obj = LR_SGD(X, y, 10, 0.1)
    beta = lr_obj.train()
    print(beta)  # [2.60151767 0.54183088]

    example_w, example_b = solution1(X, y)
    print(example_w)
    print(example_b)


if __name__ == '__main__':
    main()
s