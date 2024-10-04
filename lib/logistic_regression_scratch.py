# Logistic Regression scratch implementation


import numpy as np


class LogisticRegressionScratch:
    def __init__(self, learning_rate: float = 5e-4, max_iter: int = 10000) -> None:
        self.__learning_rate = learning_rate
        self.__max_iter = max_iter

    def sigmoid(self, y: 'np.array') -> 'np.array':
        return 1 / (1 + np.exp(-y))

    def fit(self, X: 'np.array', y: 'np.array') -> 'LogisticRegression':
        self.coef_ = np.zeros(len(X[0]) + 1)

        X = np.hstack((X, np.ones((len(X), 1))))

        for i in range(self.__max_iter):
            pred = self.sigmoid(np.dot(X, self.coef_))
            gradient = np.dot(X.T, (pred - y))
            self.coef_ -= gradient * self.__learning_rate

        return self

    def predict(self, X: 'np.array') -> 'np.array':
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        return (self.sigmoid(np.dot(X, self.coef_)) > 0.5) * 1