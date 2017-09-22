# Framework for bootcamp

## main.py
  * This is the main framework for the Neural network
  * Optimisable hyperparameter is the number of hidden layers (n_hidden)

## loadData.py
  * loads standardised data set for testing different frameworks
  * can use sklearn standard data sets http://scikit-learn.org/stable/datasets/index.html
  * returns x_train, x_test, y_train, y_test, n_classes, n_dimensionality

## gaussianProcess.py
  * performs gaussian process on given sample set
  * ideally define your process/algo in this file
  * returns y_mean, y_std

## surrogateFunction.py
  * returns maximum value of ei based on surrogate function
  * ideally define your surrogate function in this file
  * returns next_parameter

## plotProcess.py
  * visualises algorithims hyperparameter optimisation process
