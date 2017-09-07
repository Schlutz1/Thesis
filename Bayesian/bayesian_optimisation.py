# Neupy implementation of bayesian hyperparameter optimisation

from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcess
from sklearn import datasets

from neupy import algorithms, layers
from neupy import environment
#from neupy import plots

import numpy as np
import random

environment.reproducible()

dataset = datasets.load_digits()
n_samples = dataset.target.size
n_classes = 10

# One-hot encoder
target = np.zeros((n_samples, n_classes))
target[np.arange(n_samples), dataset.target] = 1

x_train, x_test, y_train, y_test = train_test_split(
		dataset.data, target, train_size=0.7
)

def train_network(n_hidden, x_train, x_test, y_train, y_test):
		network = algorithms.Momentum(
				[
						layers.Input(64),
						layers.Relu(n_hidden),
						layers.Softmax(10),
				],

				# Randomly shuffle dataset before each
				# training epoch.
				shuffle_data=True,

				# Do not show training progress in output
				verbose=False,

				step=0.001,
				batch_size=128,
				error='categorical_crossentropy',
		)
		network.train(x_train, y_train, epochs=100)

		# Calculates categorical cross-entropy error between
		# predicted value for x_test and y_test value
		return network.prediction_error(x_test, y_test)

def vector_2d(array):
		return np.array(array).reshape((-1, 1))

def gaussian_process(x_train, y_train, x_test):

		x_train = vector_2d(x_train)
		y_train = vector_2d(y_train)
		x_test = vector_2d(x_test)

		# Train gaussian process
		gp = GaussianProcess(corr='squared_exponential',
												 theta0=1e-1, thetaL=1e-3, thetaU=1)
		gp.fit(x_train, y_train)

		# Get mean and standard deviation for each possible
		# number of hidden units
		y_mean, y_var = gp.predict(x_test, eval_MSE=True)
		y_std = np.sqrt(vector_2d(y_var))

		return y_mean, y_std

def next_parameter_by_ei(y_min, y_mean, y_std, x_choices):
		# Calculate expecte improvement from 95% confidence interval
		expected_improvement = y_min - (y_mean - 1.96 * y_std)
		expected_improvement[expected_improvement < 0] = 0

		max_index = expected_improvement.argmax()
		# Select next choice
		next_parameter = x_choices[max_index]

		return next_parameter

def hyperparam_selection(func, n_hidden_range, func_args=None, n_iter=20):
    if func_args is None:
        func_args = []

    scores = []
    parameters = []

    min_n_hidden, max_n_hidden = n_hidden_range
    n_hidden_choices = np.arange(min_n_hidden, max_n_hidden + 1)

    # To be able to perform gaussian process we need to
    # have at least 2 samples.
    n_hidden = random.randint(min_n_hidden, max_n_hidden)
    score = func(n_hidden, *func_args)

    parameters.append(n_hidden)
    scores.append(score)

    n_hidden = random.randint(min_n_hidden, max_n_hidden)


    #iterates over model to find optimal number of hidden units
    for iteration in range(2, n_iter + 1):
        score = func(n_hidden, *func_args)

        parameters.append(n_hidden)
        scores.append(score)

        y_min = min(scores)
        y_mean, y_std = gaussian_process(parameters, scores,
                                         n_hidden_choices)

        n_hidden = next_parameter_by_ei(y_min, y_mean, y_std,
                                        n_hidden_choices)

        if y_min == 0 or n_hidden in parameters:
            # Lowest expected improvement value have been achieved
            break

    min_score_index = np.argmin(scores)
    return parameters[min_score_index]

best_n_hidden = hyperparam_selection(
    train_network,
    n_hidden_range=[50, 1000],
    func_args=[x_train, x_test, y_train, y_test],
    n_iter=6,
)

print best_n_hidden
