from writeFunction import write_function
from client_model import client_model
from gaussianProcess import *
from sklearn.svm import SVC
import sklearn.metrics
import numpy as np
import optunity
import csv
import sys
import random
import math


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
	architecture = [128, 128]
	hyperparam_range = [52, 76]


	normalisation_factor, min_h_range, max_h_range, h_choices = range_handling(
			hyperparam_range
		)

	# misc
	scores, parameters, trial_number, hyperparam_array, final_score = [], [], [], [], []
	opt_iteration, optimisation_range  = 0, 10

	# bayesian optimisation
	for opt_iteration in range(1, optimisation_range + 1):  # second level iteration
		print("This is bayesian iteration: " + str(opt_iteration))

		if opt_iteration in (1, 2):
				
			hp = random.randint(
					min_h_range, max_h_range)
			hp = hp * normalisation_factor

		architecture.append(hp)
		# Trains model and returns score
		final_score = train_model(architecture, filepath)
		print("Training Accuracy:", final_score)

		scores.append(final_score)
		parameters.append(hp)
		print parameters

		if opt_iteration < 2:
			continue  # 2 samples needed for inference

		y_mean, y_std = gaussian_process(
			parameters, scores, h_choices)

		y_min = min(scores)

		hp = next_parameter_by_ei(
			y_min, y_mean, y_std, h_choices)

		if y_min == 0 or hp in parameters:
			break  # lowest expectation achieved

	max_score_index = np.argmax(scores)
	np.append(final_score, scores[max_score_index])
	np.append(hyperparam_array, parameters[max_score_index])

	print(parameters[max_score_index])  # final learning rate
	print(scores[max_score_index])

	write_function(parameters, scores)
