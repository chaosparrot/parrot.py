import time
import pyautogui
from config.config import *
import math
pyautogui.FAILSAFE = False
import joblib
import numpy as np
import copy
import torch
from lib.audio_net import TinyAudioNet

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# NOTE - This classifier is only meant to be used when trained
# There is no way to 'fit' this classifier - It needs fitted classifiers to generate the state changes
class TorchEnsembleClassifier:
    
    classifiers = {}
            
    # A list of all the available classes which will be used as a starting point
    # When a prediction is made without this map having the key, it will not be added
    classes_ = []
        
    # Initialize the classifiers and their leaf classes
    def __init__( self, classifier_map ):
        self.classifiers = {}
        self.device = torch.device('cuda')
        for index, key in enumerate(classifier_map):
            state_dict = torch.load(classifier_map[key], map_location=self.device)
            self.classes_ = state_dict['labels']            
            model = TinyAudioNet(28,len(state_dict['labels']))
            model.load_state_dict(state_dict['state_dict'])
            model.to( self.device )
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
        data_row = torch.from_numpy(np.asarray(data_row, dtype=np.float32)).to( self.device )
        totalProbabilities = []
        
        with torch.no_grad():
            type = None            
            for index in self.classifiers.keys():
                probabilities = self.classifiers[index](data_row).cpu()
                
                if( len( totalProbabilities ) == 0 ):
                    totalProbabilities = probabilities
                else:
                    for probindex, probability in enumerate( probabilities ):
                        totalProbabilities[ probindex ] = totalProbabilities[ probindex ] + probability

            # Normalize the model
            for probindex, probability in enumerate( totalProbabilities ):
                totalProbabilities[ probindex ] = totalProbabilities[ probindex ] * ( 1 / len( self.classifiers.keys() ) )
                                                
        return np.asarray( totalProbabilities, dtype=np.float64 )