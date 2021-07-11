import time
import pyautogui
from config.config import *
import math
pyautogui.FAILSAFE = False
import joblib
import numpy as np
import copy
import torch
from lib.audio_net import TinyAudioNet, TinyAudioNetEnsemble, TinyRecurrent

torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# NOTE - This classifier is only meant to be used when trained
# There is no way to 'fit' this classifier - It needs fitted classifiers to generate the state changes
class TorchEnsembleClassifier:
    
    classifiers = {}
    combinedClassifier = None
            
    # A list of all the available classes which will be used as a starting point
    # When a prediction is made without this map having the key, it will not be added
    classes_ = []
        
    # Initialize the classifiers and their leaf classes
    def __init__( self, classifier_map, input_size=120, rnn=True ):
        self.classifiers = {}
        self.device = torch.device('cpu')
        classifierArray = []
        for index, key in enumerate(classifier_map):
            state_dict = torch.load(classifier_map[key], map_location=self.device)
            self.classes_ = state_dict['labels']
            if ('input_size' in state_dict):
                input_size = state_dict['input_size']
            model = TinyAudioNet(input_size, len(state_dict['labels'])) if rnn == False else TinyRecurrent(input_size, len(state_dict['labels']))
            model.load_state_dict(state_dict['state_dict'])
            model.to( self.device )
            model.double()
            model.eval()
            self.classifiers[key] = model
            classifierArray.append( key )
        self.combinedClassifier = TinyAudioNetEnsemble( list(self.classifiers.values()) )
        self.combinedClassifier.eval()
        self.combinedClassifier.to( self.device )
                                    
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
        #reshaped_input = np.reshape(data_row, (-1, 13))
        data_row = torch.from_numpy(np.asarray([data_row])).double()#.unsqueeze(0)
        data_row = data_row.to( self.device )
        totalProbabilities = []
        
        with torch.no_grad():
            type = None
            totalProbabilities = self.combinedClassifier( data_row ).cpu()
            
        return np.asarray( totalProbabilities[0], dtype=np.float64 )