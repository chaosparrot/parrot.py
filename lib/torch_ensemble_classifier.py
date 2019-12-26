import time
import pyautogui
from config.config import *
import math
pyautogui.FAILSAFE = False
import joblib
import numpy as np
import copy
import torch
from lib.audio_net import AudioNet

# NOTE - This classifier is only meant to be used when trained
# There is no way to 'fit' this classifier - It needs fitted classifiers to generate the state changes
class TorchEnsembleClassifier:
    
    classifiers = {}
            
    # A list of all the available classes which will be used as a starting point
    # When a prediction is made without this map having the key, it will not be added
    classes_ = []
        
    # Initialize the classifiers and their leaf classes
    def __init__( self, classifier_map, labels ):
        self.classes_ = labels
        self.classifiers = {}
        device = torch.device('cpu')
        for index, key in enumerate(classifier_map):
            state_dict = torch.load(classifier_map[key], map_location=device)
            model = AudioNet(28,19)
            model.load_state_dict(state_dict)
            model.eval()
            self.classifiers[key] = model        
                                    
    # Predict the probabilities of the given data array
    def predict_proba( self, data ):
        predictions = []
        for data_row in data:
            predictions.append( self.predict_single_proba(data_row) )
                
        return np.asarray( predictions )
            
    # Predict a single data row
    # This will ask all the classifiers for a prediction
    # The one with the highest prediction wins
    def predict_single_proba( self, data_row ):
        data_row = torch.tensor(data_row).float()
        totalProbabilities = []
        
        with torch.no_grad():            
            type = None
            for index in self.classifiers.keys():
                probabilities = self.classifiers[index](data_row)
                
                if( len( totalProbabilities ) == 0 ):
                    totalProbabilities = probabilities
                else:
                    for probindex, probability in enumerate( probabilities ):
                        totalProbabilities[ probindex ] = totalProbabilities[ probindex ] + probability

            # Normalize the model
            for probindex, probability in enumerate( totalProbabilities ):
                totalProbabilities[ probindex ] = totalProbabilities[ probindex ] * ( 1 / len( self.classifiers.keys() ) )
                                                
        return np.asarray( totalProbabilities, dtype=np.float64 )