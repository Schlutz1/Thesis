import csv
from datetime import datetime

def writeFunction(outputCSV, xArray, yArray, interpolBool) :
	if interpolBool == False:
		outputString = outputCSV[:-4] + str(datetime.now()) + ".csv"
		outputString = outputString.replace(':', '_')
		with open(outputString,'wb') as fin: 
				testArray = (xArray, yArray)
				fieldNames = ['dttm', 'value']
				writer = csv.writer(fin)
				testArray = zip(*testArray)
				writer.writerow(fieldNames)
				for i in range(len(xArray)):
					writer.writerow(testArray[i])
	else :
		with open(outputCSV,'wb') as fin: 
				testArray = (xArray, yArray)
				fieldNames = ['dttm', 'value']
				writer = csv.writer(fin)
				testArray = zip(*testArray)
				writer.writerow(fieldNames)
				for i in range(len(xArray)):
					writer.writerow(testArray[i])