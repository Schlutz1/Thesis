import csv
import codecs
import cStringIO

import numpy as np
import matplotlib.pyplot as plt

from writeFunction import writeFunction

#define useful params

def interpolateFunction(inputCSV, outputCSV) :

	amountOfDataPoints = 500
	timeData = []
	yData = []
	interpolateData = np.linspace(0, 1, amountOfDataPoints)

	with open (inputCSV, 'rb') as fin:

		#open csv for reading
		reader = csv.reader(fin)
		for row in reader :
			timeData.append(row[:1]) #parse dttm data
			yData.append(row[1:]) #prase value data

		yData = np.array(yData[1:]) #pass to numpy array
		
		#check if data needs to be interpolated, then interpolate
		if len(yData) < 500 : 
			yData = yData.astype(np.float) #convert data to float and interpolate
			yData = np.delete(yData, 2)
			indexData = np.linspace(0, 1, len(yData))
			yinterp = np.interp(interpolateData, indexData, yData)

			#time data mod
			timeData = np.array(timeData[1:]) #convert to numpy and mod
			timeData = np.delete(timeData, 2)
			#print timeData
			timeDataFinal = [0]*amountOfDataPoints
			netCounter = 0

			modValue = int(amountOfDataPoints / len(yData)) +1 #fix this, fucks time scale totally
			#finds new x values
			for i in range(len(timeDataFinal)) :
				timeDataFinal[i] = timeData[netCounter]
				i +=1
				if i%modValue == 0 :
					netCounter += 1
					#print netCounter
	#print file
	interpolBool = True
	writeFunction(outputCSV, timeDataFinal, yinterp, interpolBool)

	return timeDataFinal, yinterp

	

