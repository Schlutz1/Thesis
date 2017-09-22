#write results to file

import csv
import random

def write_function(trial_number, learning_rate, reward_decay, final_score) :
	id = random.randint(1, 1000)
	outputString = "PSO_meta_analysis_ " + str(id) + ".csv"
	with open(outputString,'wb') as fin: 
			testArray = (trial_number, learning_rate, reward_decay, final_score)
			fieldNames = ['trial_number', 'learning_rate', 'reward_decay', 'final_score']
			writer = csv.writer(fin)
			testArray = zip(*testArray)
			writer.writerow(fieldNames)
			for i in range(len(trial_number)):
				writer.writerow(testArray[i])