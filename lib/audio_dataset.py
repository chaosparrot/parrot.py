import torch
from torch.utils.data import Dataset, DataLoader
import os
from lib.machinelearning import *
import numpy as np
import random


class AudioDataset(Dataset):
    samples = []
    augmented_samples = []
    paths = []
    length = 0
    training = False

    def __init__(self, basedir, paths):
        self.paths = paths
		
        for index,path in enumerate(paths):
            totalpath = os.path.join(basedir,path)
            print( "Loading in " + path )
            for file in os.listdir(totalpath):
                if( file.endswith(".wav") ):
                    full_filename = os.path.join(totalpath, file)
                    #reshaped_input = np.reshape(self.feature_engineering_cached(full_filename), (-1, 13))
                    self.samples.append([full_filename, index, torch.tensor(self.feature_engineering_cached(full_filename)).float()])
                    self.augmented_samples.append([full_filename, index, torch.tensor(self.feature_engineering_augmented(full_filename)).float()])

    def set_training(self, training):
        self.training = training

    def feature_engineering_cached(self, filename):
        cached_filename = filename + "mf";
        if (os.path.isfile(cached_filename) == False ):
            data_row = training_feature_engineering(filename)
            np.savetxt( cached_filename, data_row )
        
        return np.loadtxt( cached_filename, dtype='float' )
        
    def feature_engineering_augmented(self, filename):
        return augmented_feature_engineering(filename)

                    
    def __len__(self):
        return len( self.samples )

    def __getitem__(self, idx):
        if (self.training ):
            return self.augmented_samples[idx][2], self.augmented_samples[idx][1]
        else :
            return self.samples[idx][2], self.samples[idx][1]
		
    def get_labels(self):
        return self.paths
