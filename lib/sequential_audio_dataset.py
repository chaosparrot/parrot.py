import torch
import numpy as np
from torch.utils.data import Dataset
from lib.machinelearning import *

INTERNAL_STREAM_KEY = "__streams"
class SequentialAudioDataset(Dataset):

    def __init__(self, pytorch_data):
        self.streams = pytorch_data["data"][INTERNAL_STREAM_KEY]
        self.augmented_streams = pytorch_data["augmented"][INTERNAL_STREAM_KEY]
        self.labels = list( pytorch_data["data"].keys() )
        self.labels.remove(INTERNAL_STREAM_KEY)

        self.samples = []
        self.augmented_samples = []
        self.training = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = torch.Generator(device=self.device)
        self.random_tensor = torch.tensor(1.0, requires_grad=False, device=self.device)

        background_index = self.labels.index(BACKGROUND_LABEL)        
        for index, label in enumerate( self.labels ):
            print( "Indexing " + label + "..." )
            for sample in pytorch_data["data"][label]:
                # Replace the string variant of labels to the index used
                numbered_sample = []
                for sample_frame in sample:
                    numbered_sample.append([sample_frame[0], sample_frame[1], index if sample_frame[2] == label else background_index])
                self.samples.append(numbered_sample)
            for augmented_sample in pytorch_data["augmented"][label]:

                # Replace the string variant of labels to the index used
                numbered_sample = []
                for sample_frame in augmented_sample:
                    numbered_sample.append([sample_frame[0], sample_frame[1], index if sample_frame[2] == label else background_index])
                self.augmented_samples.append(numbered_sample)

    def set_training(self, training):
        self.training = training

    def __len__(self):
        return len( self.samples )

    def __getitem__(self, idx):
        # During training, get a 10% probability that you get an augmented sample
        if self.training:
            self.random_tensor.uniform_(0, 1, generator=self.generator)
            if self.random_tensor.item() >= 0.9:
                self.transform_item_to_stream(self.augmented_samples[idx], self.augmented_streams)
        return self.transform_item_to_stream(self.samples[idx], self.streams) 

    def transform_item_to_stream(self, sample, streams):
        item = []
        idx_tags = []
        for sample_frame in sample:
            item.append(streams[sample_frame[0]][sample_frame[1]]) 
            idx_tags.append(sample_frame[2])

        #labels = torch.from_numpy(np.array(idx_tags, dtype=np.int16))
        torch_item = torch.stack(item)
        return torch_item, torch.tensor(idx_tags[-1])

    def get_labels(self):
        return self.labels