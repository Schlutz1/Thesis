import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import os, sys

path_to_file = str(sys.argv[1])


# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv(path_to_file, dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = LogisticRegression(C=25.0, dual=False)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

classification_check = np.in1d(results, testing_target)

for i in range(len(classification_check)) :
  if classification_check[i] != True :
    print "fuck up at index %i" % i
