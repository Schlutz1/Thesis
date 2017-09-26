import csv
import optunity
import sklearn.metrics
from sklearn.svm import SVC
import numpy as np

def load_csv(filename):
    data = []

    # open file
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        # load as a list of instances
        for row in reader:
            data.append(row)
    # done!
    return data

def load_labels(filename):
    data = []

    # open file
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        # load as a list of instances
        for row in reader:
            data.append(str(row[0]))
    # done!
    return data

# optimizae
# p: run in parallel?
def optimize(p, n_evals, **kwargs):
    if p:
        return 0
    else:
        return optimize_optunity(n_evals, **kwargs)

def optimize_optunity(n_evals, **kwargs):
    return optunity.maximize(score_model, num_evals=n_evals, **kwargs)

data = load_csv("abalone.data")
labels = load_labels("abalone.labels")

@optunity.cross_validated(x=data, y=labels, num_folds=2, num_iter=2)
def score_model(x_train, y_train, x_test, y_test, **params):
    model_inst = model(**params)
    model_inst.fit(x_train, y_train)
    decision_vals = model_inst.predict(x_test)
    scoring = getattr(sklearn.metrics, scoring_method)
    score = scoring(y_test, decision_vals)
    return score

model = SVC
scoring_method = "accuracy_score"
optimize(False, 5, C=[5, 7], gamma=[0, 1])