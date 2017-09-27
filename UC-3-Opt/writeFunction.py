#write results to file

import csv

def write_function(final_test_accuracy, final_train_accuracy, delta_t, architecture_range):
	with open("client data",'wb') as fin:
		csv.writer(fin)
		writer = csv.writer(fin)
		writer.writerow((final_test_accuracy, final_train_accuracy, delta_t, architecture_range))