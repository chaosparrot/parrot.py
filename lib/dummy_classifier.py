import time
import numpy as np
import copy

# NOTE - This classifier cannot be fit - It only supports silence and noise as categories
class DummyClassifier:
            
    classes_ = ['silence', 'noise']
                                    
    # Predict the probabilities of the given data array
    def predict_proba( self, data ):
        predictions = []
        for data_row in data:
            predictions.append( self.predict_single_proba(data_row) )
                
        return np.asarray( predictions )
            
    # Predict a single data row
    def predict_single_proba( self, data_row ):
        totalProbabilities = [0, 0]
        
        ## This retrieves and checks the last column of the data row - In this case, the intensity
        if( data_row[ len(data_row) - 1 ] > 1000 ):
            totalProbabilities[1] = 1
        else:
            totalProbabilities[0] = 1
        
        return np.asarray( totalProbabilities, dtype=np.float64 )