import sys
import csv
import optunity
import numpy as np
import sklearn.metrics
from sklearn.svm import SVC
from client_model import client_model
from writeFunction import write_function

#filepath = str(sys.argv[1])

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

def optimize_optunity(architecture_range,n_evals):
    #print("Arch: ", architecture)
    #print("Filename: ", filepath)
    print('Arch: ', architecture_range)
    print('N_evals: ', n_evals)
    #return client_model(architecture, filepath)
    return optunity.maximize(client_model, num_evals=n_evals, architecture_range=architecture_range)


if __name__ == "__main__" :

    n_hidden_layer, n_input_layer, n_output_layer = 0,0,0
    final_test_accuracy, final_train_accuracy, delta_t, architecture_range = None, None, None, None
    #model= scoreModel(SVC, 'accuracy_score')
    #architecture_range = {n_input_layer:[1,200],n_hidden_layer:[1,200],n_output_layer:[1,200]}
    n_evals = 1
    architecture_range = [52, 76]

    final_test_accuracy, final_train_accuracy, delta_t, architecture_range = optimize_optunity(architecture_range, n_evals)
        #solution, details, suggestion = optunity.maximize(optimize_optunity, num_evals=20, solver_name="particle swarm", architecture=[128, 128, 64])
    print("Training Accuracy:", final_score)
    print("Arch: ", architecture_range)

    write_function(final_test_accuracy, final_train_accuracy, delta_t, architecture_range)
