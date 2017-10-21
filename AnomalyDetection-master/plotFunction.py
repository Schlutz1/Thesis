import numpy as np
import matplotlib.pyplot as plt

def myPlotFunction(inputArray, anomalyArray, inputThreshold) :

	inputArray = np.array(inputArray)
	anomalyArray = np.array(anomalyArray)

	percentage = float(inputThreshold/100)
	finalLength = int(len(anomalyArray) * percentage) #array formatting, reduces noise

	anomalyArray = anomalyArray[finalLength:]

	plt.subplot(211)
	plt.plot(inputArray, 'b')
	plt.subplot(212)
	plt.plot(anomalyArray, 'r')
	plt.show()