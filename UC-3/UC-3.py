from client_model import client_model
from gaussianProcess import *
from sklearn.svm import SVC
import sklearn.metrics
import numpy as np
import optunity
import csv
import sys


filepath = str(sys.argv[1])


def train_gaussian(parameters, score, hyperparam_choices):
	# TODO
	return 0


def train_model(architecture, filepath):  # calls client model
	print("Arch: ", architecture)
	print("Filename: ", filepath)
	return client_model(architecture, filepath)
	# return optunity.maximize(scoreModel, num_evals=n_evals, **kwargs)


if __name__ == "__main__":

	# Hyperparameter definition
	architecture = [128, 128, 64]

	# misc
	scores, parameters, trial_number, hyperparam_array, final_score = [], [], [], [], []
	opt_iteration, optimisation_range, hyperparam_choices = 0, 1, 0

	# bayesian optimisation
	for opt_iteration in range(1, optimisation_range + 1):  # second level iteration
		print("This is bayesian iteration: " + str(opt_iteration))

		# Trains model and returns score
		final_score = train_model(architecture, filepath)
		print("Training Accuracy:", final_score)

		scores.append(final_score)
		parameters.append(architecture)

		if opt_iteration < 2:
			continue  # 2 samples needed for inference

		y_mean, y_std = gaussian_process(
			parameters, scores, hyperparam_choices)

		y_min = min(scores)

		learning_rate = next_parameter_by_ei(
			y_min, y_mean, y_std, hyperparam_choices)

		if y_min == 0 or learning_rate in parameters:
			break  # lowest expectation achieved

	max_score_index = np.argmax(scores)
	final_score.append(scores[max_score_index])
	learning_rate_array.append(parameters[max_score_index])

	print(parameters[max_score_index])  # final learning rate

	optimal_learning_rate = max(set(parameters), key=parameters.count)
	print("Optimal learning rate : " + str(optimal_learning_rate))
