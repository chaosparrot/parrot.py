import torch
from torch.utils.data import Dataset, DataLoader
import os
from lib.machinelearning import *

class AudioDataset(Dataset):
    samples = []
    length = 0

    def __init__(self, basedir, paths):
        self.paths = paths
		
        for path in paths:
            totalpath = os.path.join(basedir,path)
            for file in os.listdir(totalpath):
                if( file.endswith(".wav") ):
                    full_filename = os.path.join(totalpath, file)
                    self.samples.append([full_filename,path])

    def __len__(self):
        return len( self.samples )

    def __getitem__(self, idx):
        filename = self.samples[idx][0]
        data_row, frequency = feature_engineering(filename)
        return data_row, self.samples[idx][1]
