import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os
from lib.machinelearning import *
import numpy as np
import torch.optim as optim

class AudioNet(nn.Module):

    def __init__(self, inputsize, outputsize):
        super(AudioNet, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(inputsize, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, outputsize)
		
    def forward(self, x):
        x = self.relu( self.fc1(x) )
        x = self.relu( self.fc2(x) )
        x = self.relu( self.fc3(x) )
        x = self.relu( self.fc4(x) )
        x = self.fc5(x)
        if( self.training ):
            return self.log_softmax(x)
        else:
            return self.softmax(x)