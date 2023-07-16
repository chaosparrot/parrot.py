import torch
from torch.utils.data import Dataset, DataLoader
import os
from lib.machinelearning import *
import numpy as np
import random
import math
from lib.wav import load_wav_data_from_srt

class AudioDataset(Dataset):

    def __init__(self, grouped_data_directories, settings):
        self.paths = list( grouped_data_directories.keys() )
        self.settings = settings
        self.samples = []
        self.augmented_samples = []
        self.length = 0
        self.training = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = torch.Generator(device=self.device)
        self.random_tensor = torch.tensor(1.0, requires_grad=False, device=self.device)

        for index, label in enumerate( grouped_data_directories ):
            directories = grouped_data_directories[ label ]

            listed_files = {}
            for directory in directories:
                segments_directory = os.path.join(directory, "segments")
                source_directory = os.path.join(directory, "source")                
                if not (os.path.exists(segments_directory) and os.path.exists(source_directory)):
                    continue
                
                source_files = os.listdir(source_directory)
                srt_files = [x for x in os.listdir(segments_directory) if x.endswith(".srt")]
                for source_file in source_files:
                    shared_key = source_file.replace(".wav", "")
                    
                    possible_srt_files = [x for x in srt_files if x.startswith(shared_key)]
                    if len(possible_srt_files) == 0:
                        continue
                        
                    # Find the highest version of the segmentation for this source file
                    srt_file = possible_srt_files[0]
                    for possible_srt_file in possible_srt_files:
                        current_version = int( srt_file.replace(".srt", "").replace(shared_key + ".v", "") )
                        version = int( possible_srt_file.replace(".srt", "").replace(shared_key + ".v", "") )
                        if version > current_version:
                            srt_file = possible_srt_file
                    
                    listed_files[os.path.join(source_directory, source_file)] = os.path.join(segments_directory, srt_file)
            listed_files_size = len( listed_files )

            print( f"Loading in {label}" )
            listed_source_files = listed_files.keys()
            for file_index, full_filename in enumerate( listed_source_files ):
                all_samples = load_wav_data_from_srt(listed_files[full_filename], full_filename, self.settings['FEATURE_ENGINEERING_TYPE'], False)
                augmented_samples = load_wav_data_from_srt(listed_files[full_filename], full_filename, self.settings['FEATURE_ENGINEERING_TYPE'], False, True)
                
                for sample in all_samples:
                    self.samples.append([full_filename, index, torch.tensor(sample).float()])
                for augmented_sample in augmented_samples:
                    self.augmented_samples.append([full_filename, index, torch.tensor(augmented_sample).float()])

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
