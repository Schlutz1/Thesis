from pprint import pprint
from functools import partial

import theano
import numpy as np
from sklearn import model_selection, datasets, preprocessing, metrics
import hyperopt
from hyperopt import hp
from neupy import algorithms, layers, environment
from neupy.exceptions import StopTraining

theano.config.floatX = 'float32'

def on_epoch_end(network):
    if network.errors.last() > 10:
        raise StopTraining("Training was interrupted. Error is to high.")

def train_network(parameters):
    print("Parameters:")
    pprint(parameters)
    print()

    step = parameters['step']
    batch_size = int(parameters['batch_size'])
    proba = parameters['dropout']
    activation_layer = parameters['act_func_type']
    layer_sizes = [int(n) for n in parameters['layers']['n_units_layer']]

    network = layers.Input(784)

    for layer_size in layer_sizes:
        network = network > activation_layer(layer_size)

    network = network > layers.Dropout(proba) > layers.Softmax(10)

    mnet = algorithms.RMSProp(
        network,

        batch_size=batch_size,
        step=step,

        error='categorical_crossentropy',
        shuffle_data=True,

        epoch_end_signal=on_epoch_end,
    )
    mnet.train(x_train, y_train, epochs=50)

    score = mnet.prediction_error(x_test, y_test)

    y_predicted = mnet.predict(x_test).argmax(axis=1)
    accuracy = metrics.accuracy_score(y_test.argmax(axis=1), y_predicted)

    print("Final score: {}".format(score))
    print("Accuracy: {:.2%}".format(accuracy))

    return score


def uniform_int(name, lower, upper):
    # `quniform` returns:
    # round(uniform(low, high) / q) * q
    return hp.quniform(name, lower, upper, q=1)

def loguniform_int(name, lower, upper):
    # Do not forget to make a logarithm for the
    # lower and upper bounds.
    return hp.qloguniform(name, np.log(lower), np.log(upper), q=1)

def load_mnist_dataset():
    mnist = datasets.fetch_mldata('MNIST original')

    target_scaler = preprocessing.OneHotEncoder()
    target = mnist.target.reshape((-1, 1))
    target = target_scaler.fit_transform(target).todense()

    data = mnist.data / 255.
    data = data - data.mean(axis=0)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        data.astype(np.float32),
        target.astype(np.float32),
        train_size=(6 / 7.)
    )

    return x_train, x_test, y_train, y_test

environment.reproducible()
x_train, x_test, y_train, y_test = load_mnist_dataset()

# Object stores all information about each trial.
# Also, it stores information about the best trial.
trials = hyperopt.Trials()

parameter_space = {
    'step': hp.uniform('step', 0.01, 0.5),
    'layers': hp.choice('layers', [{
        'n_layers': 1,
        'n_units_layer': [
            uniform_int('n_units_layer_11', 50, 500),
        ],
    }, {
        'n_layers': 2,
        'n_units_layer': [
            uniform_int('n_units_layer_21', 50, 500),
            uniform_int('n_units_layer_22', 50, 500),
        ],
    }]),
    'act_func_type': hp.choice('act_func_type', [
        layers.Relu,
        layers.PRelu,
        layers.Elu,
        layers.Tanh,
        layers.Sigmoid
    ]),

    'dropout': hp.uniform('dropout', 0, 0.5),
    'batch_size': loguniform_int('batch_size', 16, 512),
}

tpe = partial(
    hyperopt.tpe.suggest,

    # Sample 1000 candidate and select candidate that
    # has highest Expected Improvement (EI)
    n_EI_candidates=1000,

    # Use 20% of best observations to estimate next
    # set of parameters
    gamma=0.2,

    # First 20 trials are going to be random
    n_startup_jobs=20,
)

hyperopt.fmin(
    train_network,
    trials=trials,
    space=parameter_space,

    # Set up TPE for hyperparameter optimization
    algo=tpe,

    # Maximum number of iterations. Basically it trains at
    # most 200 networks before choose the best one.
    max_evals=200,
)
