#class encoding file
import pandas as pd
import numpy as np

import csv, sys, os

filename = str(sys.argv[1])

def openFunction() :
	csv_data = pd.read_csv(filename)
	csv_data['class'].apply(pd.to_numeric, errors='coerce')
	print csv_data['class']


if __name__ == "__main__" :
	openFunction()
