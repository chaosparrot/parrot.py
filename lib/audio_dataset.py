import torch
from torch.utils.data import Dataset
from lib.machinelearning import *

class AudioDataset(Dataset):

    def __init__(self, pytorch_data):
        self.paths = list( pytorch_data["data"].keys() )
        self.samples = []
        self.augmented_samples = []
        self.length = 0
        self.training = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = torch.Generator(device=self.device)
        self.random_tensor = torch.tensor(1.0, requires_grad=False, device=self.device)
        
        for index, label in enumerate( pytorch_data["data"] ):
            print( "Indexing " + label + "..." )
            for sample in pytorch_data["data"][label]:
                self.samples.append([sample[0], index, sample[1]])
            for augmented_sample in pytorch_data["augmented"][label]:
                self.augmented_samples.append([augmented_sample[0], index, augmented_sample[1]])            

    def set_training(self, training):
        self.training = training

    def __len__(self):
        return len( self.samples )

    def __getitem__(self, idx):
        # During training, get a 10% probability that you get an augmented sample
        if self.training:
            self.random_tensor.uniform_(0, 1, generator=self.generator)
            if self.random_tensor.item() >= 0.9:
                return self.augmented_samples[idx][2], self.augmented_samples[idx][1]
        return self.samples[idx][2], self.samples[idx][1]

    def get_labels(self):
        return self.paths