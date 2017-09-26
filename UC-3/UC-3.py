import csv
import optunity
import sklearn.metrics
from sklearn.svm import SVC
import numpy as np

from client_model import *

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


class scoreModel:
    def __init__(self, model, scoring_method):
        self.model = model
        self.scoring_method = scoring_method    #str of name of scoring method to use
        
    def score_model(self, x_train, y_train, x_test, y_test, **params):
        model_inst = self.model(**params)
        model_inst.fit(x_train, y_train)
        decision_vals = model_inst.predict(x_test)
        scoring = getattr(sklearn.metrics, self.scoring_method)
        score = scoring(y_test, decision_vals)
        return score

    def hyp_opt_optunity(self, data, labels, p, n_evals, **kwargs):
        if p:
            return 0
        else:
            return self.optimize_optunity(n_evals, data, labels, **kwargs)
    
    def optimize_optunity(self, n_evals, data, labels, **kwargs):
        cv_decorator = optunity.cross_validated(x=data, y=labels, num_folds=5)
        scoreModel = cv_decorator(self.score_model)
        return optunity.maximize(scoreModel, num_evals=n_evals, **kwargs)


if __name__ == "__main__" :

    model= scoreModel(SVC, 'accuracy_score')
    model.hyp_opt_optunity(data, labels, False, 5, C=[5, 7], gamma=[0, 1])

