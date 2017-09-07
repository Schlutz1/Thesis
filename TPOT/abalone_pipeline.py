#Abalone trial


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tpot import TPOTClassifier

import pandas as pd
import numpy as np
import os, sys

mlb = MultiLabelBinarizer()

#pass in variables
path_to_file = str(sys.argv[1])

#import data and process
tpot_data = pd.read_csv(path_to_file)
X, y = tpot_data.iloc[:,:-1], tpot_data.iloc[:,-1]

X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, train_size=0.75, test_size=0.25)
X_test, X_val, y_test, y_val = train_test_split(X_holdout, y_holdout)


tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)

tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('abalone_tpot_pipeline.py')

#tpot.fit(tpot_data[training_indices], tpot_class[training_indices])
