import torch
from torch.utils.data import Dataset, DataLoader
import os
from lib.machinelearning import *
import numpy as np
import random
import math

class AudioDataset(Dataset):
    samples = []
    augmented_samples = []
    paths = []
    length = 0
    settings = {}
        
    training = False

    def __init__(self, basedir, paths, settings):
        self.paths = paths
        self.settings = settings
		
        rebuild_cache = False
        for index,path in enumerate(paths):
            totalpath = os.path.join(basedir,path)
            print( "Loading in " + path )
            listed_files = os.listdir(totalpath)
            listed_files_size = len( listed_files )
            for file_index, file in enumerate(listed_files):            
                if( file.endswith(".wav") ):
                    print( str( math.floor(((file_index + 1 ) / listed_files_size ) * 100)) + "%", end="\r" )
                    full_filename = os.path.join(totalpath, file)
                    
                    # When the input length changes due to a different input type being used, we need to rebuild the cache from scratch
                    if (index == 0 and file_index == 0):
                        rebuild_cache = len(self.feature_engineering_cached(full_filename, False)) != len(self.feature_engineering_augmented(full_filename))
                        
                    self.samples.append([full_filename, index, torch.tensor(self.feature_engineering_cached(full_filename, rebuild_cache)).float()])
                    self.augmented_samples.append(None)

    def set_training(self, training):
        self.training = training

    def feature_engineering_cached(self, filename, rebuild_cache=False):
        # Only build a filesystem cache of feature engineering results if we are dealing with non-raw wave form
        if (self.settings['FEATURE_ENGINEERING_TYPE'] != 1):
            cache_dir = os.path.join(filename, "..", "cache")
            os.makedirs(cache_dir, exist_ok=True)
            cached_filename = os.path.join(cache_dir, os.path.basename(filename) + "_fe")
            if (os.path.isfile(cached_filename) == False or rebuild_cache == True):
                data_row = training_feature_engineering(filename, self.settings)
                np.savetxt( cached_filename, data_row )
        else:
            cached_filename = filename
        
        return np.loadtxt( cached_filename, dtype='float' )
        
    def feature_engineering_augmented(self, filename):
        return augmented_feature_engineering(filename, self.settings)
                    
    def __len__(self):
        return len( self.samples )

    def __getitem__(self, idx):
        # During training, get a 10% probability that you get an augmented sample
        if (self.training and random.uniform(0, 1) >= 0.9 ):
            if (self.augmented_samples[idx] is None):
                self.augmented_samples[idx] = [self.samples[idx][0], self.samples[idx][1], torch.tensor(self.feature_engineering_augmented(self.samples[idx][0])).float()]
            return self.augmented_samples[idx][2], self.augmented_samples[idx][1]
        else:
            return self.samples[idx][2], self.samples[idx][1]
		
    def get_labels(self):
        return self.paths
