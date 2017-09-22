#main file framework

#from matplotlib import pyplot as plt
from neupy import algorithms, layers
import random

# these imports are used as the parameter optimisation
from surrogateFunction import *
from gaussianProcess import *
#from plotProcess import plot_process

# function used to load standardised data sets
from loadData import load_data

import sys

#pass in variables
for counter in range(len(sys.argv)) :
    if counter >= 1 :
        path_to_file = str(sys.argv[1]) #checks for variable amount of input args
    else :
        path_to_file = None

def train_network(n_hidden, x_train, x_test, y_train, y_test, n_classes, n_dimensionality):
    network = algorithms.Momentum(
        [
            layers.Input(n_dimensionality), #input dimensionality
            layers.Relu(n_hidden), #optimisable hyperparam
            layers.Softmax(n_classes), #class output
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
    network.train(x_train, y_train, x_test, y_test, epochs=100)

    # Calculates categorical cross-entropy error between
    # predicted value for x_test and y_test value
    return network, network.prediction_error(x_test, y_test)

def hyperparam_selection(func, n_hidden_range, func_args=None, n_iter=20):
    if func_args is None:
        func_args = []

    scores = []
    parameters = []

    min_n_hidden, max_n_hidden = n_hidden_range
    n_hidden_choices = np.arange(min_n_hidden, max_n_hidden + 1)

    for iteration in range(1, n_iter + 1):
        if iteration in (1, 2):
            n_hidden = random.randint(min_n_hidden, max_n_hidden)

        print('-----------------------')
        print('Iteration #{}'.format(iteration))
        print("Number of hidden layers: {}".format(n_hidden))

        nnet, score = func(n_hidden, *func_args)

        print("Cross entropy score: {}".format(score))

        parameters.append(n_hidden)
        scores.append(score)

        # To be able to perfome gaussian process we need to
        # have at least 2 samples.
        if iteration < 2:
            continue

        y_mean, y_std = gaussian_process(parameters, scores,
                                         n_hidden_choices)
        y_min = min(scores)

        n_hidden = next_parameter_by_ei(y_min, y_mean, y_std,
                                        n_hidden_choices)

        if y_min == 0 or n_hidden in parameters:
            # Lowest expected improvement value have been achieved
            break

    min_score_index = np.argmin(scores)
    return parameters[min_score_index]

if __name__ == '__main__':

  x_train, x_test, y_train, y_test, n_classes, n_dimensionality = load_data(path_to_file)

  #parses payload
  best_n_hidden = hyperparam_selection(
    train_network,
    n_hidden_range=[50, 1000],
    func_args=[x_train, x_test, y_train, y_test, n_classes, n_dimensionality],
    n_iter=6,
  )

  print best_n_hidden
