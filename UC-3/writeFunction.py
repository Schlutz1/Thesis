import csv
from datetime import datetime

def write_function(parameters, scores) :
	outputCSV = "data/client_optimisation.csv"
	outputString = outputCSV[:-4] + str(datetime.now()) + ".csv"
	with open(outputString,'wb') as fin: 
			testArray = (parameters, scores)
			fieldNames = ['params', 'test_accuracy']
			writer = csv.writer(fin)
			testArray = zip(*testArray)
			writer.writerow(fieldNames)
			for i in range(len(parameters)):
				writer.writerow(testArray[i])