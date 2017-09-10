
from sklearn.model_selection import train_test_split
from sklearn import datasets

from neupy import environment
import numpy as np

def load_data() :
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
  return x_train, x_test, y_train, y_test
