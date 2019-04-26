import time
import pyautogui
from config.config import *
import math
pyautogui.FAILSAFE = False
from sklearn.externals import joblib
import numpy as np
import copy

# NOTE - This classifier is only meant to be used when trained
# There is no way to 'fit' this classifier - It needs fitted classifiers to generate the tree
class HierarchialClassifier:

	# A map of all the classifiers inside of this classifier
	classifiers = {}

	# A list of all the available classes which will be used as a starting point
	# When a prediction is made without this map having the key, it will not be added
	classes_ = []
		
	# Initialize the classifiers and their leaf classes
	def __init__( self, classifier_map ):
		self.classes_ = []
		self.classifiers = classifier_map
		
		for index,classifier_label in enumerate( self.classifiers ):
			for label in self.classifiers[ classifier_label ].classes_:
				if( label not in self.classifiers ):
					self.classes_.append( label )
					
	# Predict the probabilities of the given data array
	def predict_proba( self, data ):
		predictions = []
		for data_row in data:
			predictions.append( self.predict_single_proba(data_row) )
		
		return np.asarray( predictions )
		
	# Predict a single data row
	# This will recursively go through the tree structure of classifiers until it has reached an end
	# When it reaches the end, it will only retain the end nodes probabilities since that classifier is deemed to be the most knowledgeable
	def predict_single_proba( self, data_row, type="main" ):
		probabilityList = []
		probabilities = self.classifiers[ type ].predict_proba( [data_row] )[0]
		
		predicted = np.argmax( probabilities )
		if( isinstance(predicted, list) ):
			predicted = predicted[0]
		
		predicted_label = self.classifiers[type].classes_[ predicted ]
		
		# Check if the winner is actually another category that needs to be classified
		if( predicted_label in self.classifiers.keys() ):
			return self.predict_single_proba( data_row, predicted_label )
		
		# Leaf node classifier
		else:
			probabilityDict = {}
			for index, percent in enumerate( probabilities ):
				label = self.classifiers[ type ].classes_[ index ]
				probabilityDict[ label ] = percent

			# Make the complete list of probabilities in the order of the classes array
			for label in self.classes_:
				if( label in probabilityDict ):
					probabilityList.append( probabilityDict[label] )
				else:
					probabilityList.append( 0 )
					
		return np.asarray( probabilityList, dtype=np.float64 )