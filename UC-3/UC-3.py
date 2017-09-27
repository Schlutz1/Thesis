import sys
import csv
import optunity
import numpy as np
import sklearn.metrics
from sklearn.svm import SVC
from client_model import client_model

filepath = str(sys.argv[1])

'''
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
'''

def optimize_optunity(architecture, filepath):
    print("Arch: ", architecture)
    print("Filename: ", filepath)
    return client_model(architecture, filepath)
    #return optunity.maximize(scoreModel, num_evals=n_evals, **kwargs)


if __name__ == "__main__" :

    #model= scoreModel(SVC, 'accuracy_score')
    architecture = [128, 128, 64]
    final_score = optimize_optunity(architecture, filepath)
    #solution, details, suggestion = optunity.maximize(optimize_optunity, num_evals=20, solver_name="particle swarm", architecture=[128, 128, 64])
    print("Training Accuracy:", final_score)

