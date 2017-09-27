from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np


def next_parameter_by_ei(y_min, y_mean, y_std, x_choices):
    # Calculate expecte improvement from 95% confidence interval
    expected_improvement = y_min - (y_mean - 1.96 * y_std)
    expected_improvement[expected_improvement < 0] = 0

    max_index = expected_improvement.argmax()
    # Select next choice
    next_parameter = x_choices[max_index]

    return next_parameter


def range_handling(hyperpameter_range):
    min_hyperparam, max_hyperparam = hyperpameter_range
    hyperparam_choices = np.arange(min_hyperparam, max_hyperparam + 1)

    # needed in cases where value is float
    normalisation_factor = min_hyperparam
    min_hyperparam, max_hyperparam = min_hyperparam / \
        normalisation_factor, max_hyperparam / normalisation_factor

    return normalisation_factor, min_hyperparam, max_hyperparam, hyperparam_choices


def vector_2d(array):
    return np.array(array).reshape((-1, 1))


def gaussian_process(x_train, y_train, x_test):
    x_train = vector_2d(x_train)
    y_train = vector_2d(y_train)
    x_test = vector_2d(x_test)

    # Train gaussian process
    # adjust kernel for better results?
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(x_train, y_train)

    # Get mean and standard deviation for each possible
    # number of hidden units
    y_mean, y_var = gp.predict(x_test, return_std=True)
    y_std = np.sqrt(vector_2d(y_var))

    return y_mean, y_std
