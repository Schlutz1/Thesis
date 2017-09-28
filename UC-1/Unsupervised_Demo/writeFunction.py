#write results to file

import csv

def clear_file(filename):
	open(filename, 'w').close()
	write_function(filename, "learn_type", "algo_type", "model", "num_hp",
		"eval_type", "iteration", "Learning Rate", "Reward Decay", "eval_score")

def write_function(filename, learn_type, algo_type, model, num_hp, eval_type,\
	iteration, hp1, hp2, eval_score):
	with open(filename,'a') as fin:
		csv.writer(fin)
		writer = csv.writer(fin)
		writer.writerow((learn_type, algo_type, model, num_hp, eval_type, iteration, eval_score, hp1, hp2))