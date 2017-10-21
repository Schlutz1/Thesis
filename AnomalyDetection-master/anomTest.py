# test function for anom test stuff

import sys
import csv
import math
import datetime
import dateutil.parser
import json

#import matplotlib.pyplot as plt
import numpy as np

from pprint import pprint
from optparse import OptionParser

from nupic.frameworks.opf.modelfactory import ModelFactory
from nupic.frameworks.opf.predictionmetricsmanager import MetricsManager
from nupic.algorithms.anomaly_likelihood import AnomalyLikelihood

from plotFunction import myPlotFunction
from interpolateFunction import interpolateFunction
from writeFunction import writeFunction

#global variables
#inputFileNameInterpol = "data\\SampleDataTwo.csv"
inputFileNameInterpol = str(sys.argv[1])
inputFileNameLocal = "data\\SplunkDataInterpol.csv"
#inputFileNameLocal = str(sys.argv[2])
outputFileName = "data\\SplunkFinalData.csv"

def runAnomaly(options) :

	#define local params :

	inputArray = [] #holds all input data
	anomalyArray = [] #holds all output data
	inputThreshold = float(10) #how many percent of intial samples to ignore
	anomCounter = 0 #counts number of anomalies

	[timeDataFinal, yvalues] = interpolateFunction(inputFileNameInterpol, inputFileNameLocal) #interpolate the function


	with open("model_params.json") as fp:
		modelParams = json.load(fp)
		#pprint(modelParams)

	#JSON handling
	sensorParams = modelParams['modelParams']['sensorParams']
	numBuckets = modelParams['modelParams']['sensorParams']['encoders']['value'].pop('numBuckets')
	#print numBuckets
	resolution = options.resolution

	#fuck is resolution
	if resolution is None:
		resolution = max(0.001, (options.max - options.min) / numBuckets)
		print "Using resolution value: {0}".format(resolution)
	sensorParams['encoders']['value']['resolution'] = resolution
	#print resolution

	model = ModelFactory.create(modelParams)
	model.enableInference({'predictedField': 'value'})
	with open (options.inputFile) as fin:

		#Open files
		#Setup headers
		reader = csv.reader(fin)
		headers = reader.next()
	
		# The anomaly likelihood object
		anomalyLikelihood = AnomalyLikelihood()

		#Iterate through each record in the CSV
		print "Starting processing at", datetime.datetime.now()
		for i, record in enumerate(reader, start=1):

			# Convert input data to a dict so we can pass it into the model
			inputData = dict(zip(headers, record))
			#print(inputData)
			inputData["value"] = float(inputData["value"])
			inputArray.append(inputData["value"])
			inputData["dttm"] = dateutil.parser.parse(inputData["dttm"])
			#print inputData

			# Send it to the CLA and get back the raw anomaly score
			result = model.run(inputData)

			#inferences call from nupic
			anomalyScore = result.inferences['anomalyScore']
			anomalyArray.append(anomalyScore)

			#comput likelihood - nupic call
			likelihood = anomalyLikelihood.anomalyProbability(inputData["value"], anomalyScore, inputData["dttm"])


		myPlotFunction(inputArray, anomalyArray, inputThreshold) #plot the output

		#print file
		interpolBool = False
		writeFunction(outputFileName, timeDataFinal, anomalyArray, interpolBool)

if __name__ == '__main__':

	helpString = (
		"\n%prog [options] [uid]"
		"\n%prog --help"
		"\n"
		"\nRuns NuPIC anomaly detection on a csv file."
		"\nWe assume the data files have a timestamp field called 'dttm' and"
		"\na value field called 'value'. All other fields are ignored."
		"\nNote: it is important to set min and max properly according to data."
	)

	parser = OptionParser(helpString)
	parser.add_option("--inputFile",
						help="Path to data file. (default: %default)", 
						dest="inputFile", default=inputFileNameLocal)
	parser.add_option("--outputFile",
						help="Output file. Results will be written to this file."
						" (default: %default)", 
						dest="outputFile", default=outputFileName)
	parser.add_option("--max", default=100.0, type=float,
		  help="Maximum number for the value field. [default: %default]")
	parser.add_option("--min", default=0.0, type=float,
		  help="Minimum number for the value field. [default: %default]")
	parser.add_option("--resolution", default=None, type=float,
		  help="Resolution for the value field (overrides min and max). [default: %default]")
	  
	options, args = parser.parse_args(sys.argv[1:])

	  # Run it
	runAnomaly(options)
