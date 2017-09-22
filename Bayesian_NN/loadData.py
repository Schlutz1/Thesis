
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets

from neupy import environment

import pandas as pd
import numpy as np

import sys

def load_data(path_to_file) :

  environment.reproducible()

  #take str args, if none load default trial set
  if path_to_file == None :

    print("Loading sklearn dataset")
    dataset = datasets.load_digits()
    n_samples = dataset.target.size
    n_dimensionality = dataset.data.shape[1] #gives input dimensions

    n_classes = []
    for counter in dataset.target :
        if counter not in n_classes :
          n_classes.append(counter)
    n_classes = max(n_classes) + 1 #gives output dimensions

    # One-hot encoder
    target = np.zeros((n_samples, n_classes))
    target[np.arange(n_samples), dataset.target] = 1

    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data, target, train_size=0.7
    )

  #import data form custom file
  else :

    dataset = pd.read_csv(path_to_file)
    data, target_raw = dataset.iloc[:,:-1], dataset.iloc[:,-1]
    n_samples = dataset.shape[0]
    n_dimensionality = data.shape[1]

    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(target_raw)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    target = onehot_encoder.fit_transform(integer_encoded)

    n_classes = target.shape[1]

    x_train, x_test, y_train, y_test = train_test_split(
        data, target, train_size=0.7
    )

  return x_train, x_test, y_train, y_test, n_classes, n_dimensionality

#if __name__ == '__main__':
  #load_data( str(sys.argv[1]) )
