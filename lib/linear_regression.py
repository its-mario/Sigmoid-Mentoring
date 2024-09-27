import numpy as np
import math


class LinearRegressionScratch():
    #Define a Linear Regression class to store our relevant functions in
    def __init__(self):
        '''
            Initializes the Linear Regression model.
            It also stores the self.params__ variable,
            which
            will be the weights that the model returns.
        '''
        self.params__ = None
    def fit(self, X, y, learning_rate=0.00001, iterations=10, batch_size=16):
        '''
            This function applies the Gradient Descent
            model
            to the dataset
            :param X: numpy.ndarray
            The X matrix containing the independent
            variable columns.
            :param y: numpy.ndarray
            The target vector y.
        '''

        #Add a column of ones for the constant term
        # X = np.concatenate([X, np.ones_like(y)], axis = 1)
        rows, cols = X.shape

        #Combine the X and y columns to more easily shuffle it later
        X = np.append(X, y, axis=1)

        #Make the initial random guess for w
        w = np.random.random((cols, 1))

        #Go through all the iterations
        for i in range(iterations):

            #Shuffle the rows of the data
            np.random.shuffle(X)

            #Define X and y again
            y_it = X[:, -1].reshape((rows, 1))
            X_it = X[:, :-1]

            #Iterate through the batches
            for batch in range(math.ceil(rows / batch_size)):

                batch_start = batch * batch_size

                #Cut a batch from the dataset
                x_batch = X_it[batch_start : min(batch_start + batch_size, X.shape[0])]
                y_batch = y_it[batch_start : min(batch_start + batch_size, X.shape[0])]

                #Subtract the gradient from our previous estimation
                w -= learning_rate * np.matmul(x_batch.transpose(), (np.matmul(x_batch, w) - y_batch))

        self.params__= w
        return self

    def predict(self, X):
        # X = np.concatenate([X, np.ones(X.shape[0]).reshape((X.shape[0], 1))], axis=1)
        return np.matmul(X, self.params__)