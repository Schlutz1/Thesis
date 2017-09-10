
from sklearn.model_selection import train_test_split
from sklearn import datasets

from neupy import environment
import numpy as np

def load_data() :
  environment.reproducible()

  dataset = datasets.load_digits()
  n_samples = dataset.target.size
  n_dimensionality = dataset.data.shape[1] #10 digits, 3 iris

  n_classes = []
  for counter in dataset.target :
      if counter not in n_classes :
        n_classes.append(counter)
  n_classes = max(n_classes) + 1

  # One-hot encoder
  target = np.zeros((n_samples, n_classes))
  target[np.arange(n_samples), dataset.target] = 1

  x_train, x_test, y_train, y_test = train_test_split(
      dataset.data, target, train_size=0.7
  )
  return x_train, x_test, y_train, y_test, n_classes, n_dimensionality

#if __name__ == '__main__':
  #load_data()
