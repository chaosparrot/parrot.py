import time
import pyautogui
from config.config import *
import math
pyautogui.FAILSAFE = False
import joblib
import numpy as np
import copy

# NOTE - This classifier is only meant to be used when trained
# There is no way to 'fit' this classifier - It needs a fitted classifier
class ChangeResistantClassifier:
	
	# The latest prediction made
	previous_prediction = []
	latest_prediction = 0
		
	# The classifier that does all the classification
	classifier = None
	
	# A list of all the available classes which will be used as a starting point
	# When a prediction is made without this map having the key, it will not be added
	classes_ = []
		
	# Initialize the classifiers and their leaf classes
	def __init__( self, classifier_map ):
		self.classifier = classifier_map['main']
		self.classes_ = self.classifier.classes_
		
	# Predict the probabilities of the given data array
	def predict_proba( self, data ):
		predictions = []
		for data_row in data:
			predictions.append( self.predict_single_proba(data_row) )
		
		return np.asarray( predictions )
			
	# Predict a single data row
	# This classifier keeps the weights of the previous prediction and uses them if the current prediction isn't as accurate
	def predict_single_proba( self, data_row ):
		probabilityList = []
		probabilities = self.classifier.predict_proba( [data_row] )[0]
				
		prediction_time = time.time()
		apply_previous_weights = prediction_time - self.latest_prediction < 0.05
		
		if( apply_previous_weights == True ):
			for index, previous_probability in enumerate( self.previous_prediction ):
				probabilities[index] = ( probabilities[index] + ( previous_probability * 0.5 ) ) * 0.66666
						
		# Set the state for the next predictions
		self.previous_prediction = probabilities
		self.latest_prediction = prediction_time
		
		return probabilities