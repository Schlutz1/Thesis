#write results to file

import csv

def write_function(trial_number, learning_rate, final_score) :
	outputString = "cartpole_meta_analysis.csv"
	with open(outputString,'wb') as fin: 
			testArray = (trial_number, learning_rate, final_score)
			fieldNames = ['trial_number', 'learning_rate', 'final_score']
			writer = csv.writer(fin)
			testArray = zip(*testArray)
			writer.writerow(fieldNames)
			for i in range(len(trial_number)):
				writer.writerow(testArray[i])
