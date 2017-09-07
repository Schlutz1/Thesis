import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv(str(sys.argv[1]), dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = RandomForestClassifier(max_features=0.65, min_samples_leaf=15, min_samples_split=13, n_estimators=100)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

counter = 0
for i in range(len(results)) :
  if results[i] != testing_target[i] :
    print "fucked it, values are: %i and %i" % (results[i], testing_target[i])
    counter += 1

confidence_interval = ((len(results) - counter) * 100.0) / len(results)

print "Net confidence: %f" % confidence_interval
