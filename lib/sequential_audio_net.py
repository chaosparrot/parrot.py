import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os
from lib.machinelearning import *
import numpy as np
import csv
from config.config import *
import torch.optim as optim
import time 
from lib.combine_models import connect_model
from lib.key_poller import KeyPoller
import random

class TinySequentialAudioNet(nn.Module):

    def __init__(self, inputsize, outputsize, only_logsoftmax=False):
        super(TinySequentialAudioNet, self).__init__()
        self.gru_layer_dim = 1
        self.hidden_dim = 256
        self.linear_dim = 512

        self.only_logsoftmax = only_logsoftmax
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()
        self.dropOut = nn.Dropout(p=0.15)
        
        self.rnn = nn.GRU(inputsize, self.hidden_dim, self.gru_layer_dim, bidirectional=False, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, self.linear_dim)
        self.fc2 = nn.Linear(self.linear_dim, self.linear_dim)        
        self.fc3 = nn.Linear(self.linear_dim, outputsize)
        self.h0 = None
		
    def forward(self, x):
        x, h0 = self.rnn(x, self.h0)
        self.h0 = h0
        x = self.dropOut(self.relu(self.fc1(x)))
        x = self.dropOut(self.relu(self.fc2(x)))
        x = self.fc3(x)
        if( self.training or self.only_logsoftmax ):
            return self.log_softmax(x)
        else:
            return self.softmax(x)

class TinySequentialAudioNetEnsemble(nn.Module):
    def __init__(self, models):
        super(TinySequentialAudioNetEnsemble, self).__init__()
        self.models = []
        self.model_length = len(models)
        for model in models:
            self.models.append(model)
            
    def forward(self, x):
        out = 0
        for index, model in enumerate(self.models):
            if (index == 0):
                out = model(x)
            else:
                out = out + model(x)
        
        return out / self.model_length
            
class SequentialAudioNetTrainer:
    nets = []
    dataset_labels = []
    dataset_size = 0
    
    optimizers = []
    validation_loaders = []
    train_loaders = []
    criterion = nn.NLLLoss()
    batch_size = 2
    validation_split = .2
    max_epochs = 300
    random_seeds = []
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dataset = False
    train_indices = []
    input_size = 160
    
    def __init__(self, dataset, net_count = 1, audio_settings = None):
        self.net_count = net_count
        x, y = dataset[0]
        self.input_size = len(x[0])
        print("INPUT SIZE", self.input_size)
        self.dataset_labels = dataset.get_labels()
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.audio_settings = audio_settings
        self.dataset_size = len(dataset)
        
        split = int(np.floor(self.validation_split * self.dataset_size))

        for i in range(self.net_count):
            self.nets.append(TinySequentialAudioNet(self.input_size, len(self.dataset_labels), True))
            self.optimizers.append(optim.SGD(self.nets[i].parameters(), lr=0.003, momentum=0.9, nesterov=True))
            self.random_seeds.append(random.randint(0, 100000))
 
            # Split the dataset into validation and training data loaders
            indices = list(range(self.dataset_size))
            np.random.seed(self.random_seeds[i])
            np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]
            self.train_indices.append( train_indices)

            train_sampler = SubsetRandomSampler(self.train_indices[i])
            valid_sampler = SubsetRandomSampler(val_indices)
            self.train_loaders.append(torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler, pin_memory=False, num_workers=0))
            self.validation_loaders.append(torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, sampler=valid_sampler, pin_memory=False, num_workers=0))

    def train(self, filename):
        best_accuracy = []
        combined_classifier_map = {}
        for i in range(self.net_count):
            self.nets[i] = self.nets[i].to(self.device)
            combined_classifier_map['classifier_' + str(i)] = os.path.join(CLASSIFIER_FOLDER, filename + '_' + str(i + 1) + '-BEST-weights.pth.tar')
            best_accuracy.append(0)

        starttime = int(time.time())
        combined_model = TinySequentialAudioNetEnsemble(self.nets).to(self.device)
        
        input_size = 120
        with open(REPLAYS_FOLDER + "/sequential_model_training_" + filename + str(starttime) + ".csv", 'a', newline='') as csvfile:
            headers = ['epoch', 'loss', 'avg_validation_accuracy']
            headers.extend(self.dataset_labels)
            writer = csv.DictWriter(csvfile, fieldnames=headers, delimiter=',')
            writer.writeheader()
            for epoch in range(self.max_epochs):
                # Training
                self.dataset.set_training(True)
                epoch_loss = 0.0
                running_loss = []
                for j in range(self.net_count):
                    running_loss.append(0.0)
                    self.nets[j].train(True)
                    
                    i = 0
                    with torch.set_grad_enabled(True):
                        for local_batch, local_labels in self.train_loaders[j]:
                            # Transfer to GPU
                            print( local_batch )
                            print( local_labels )
                            local_labels = local_labels.to(self.device)
                            local_batch = local_batch.to(self.device)
                            
                            # Zero the gradients for this batch
                            i += 1                        
                            net = self.nets[j]
                            optimizer = self.optimizers[j]
                            optimizer.zero_grad()
                        
                            # Calculating loss
                            output = net(local_batch)
                            loss = self.criterion(output, local_labels)
                            loss.backward()
                                    
                            # Prevent exploding weights
                            torch.nn.utils.clip_grad_norm_(net.parameters(),4)
                            optimizer.step()
                        
                            running_loss[j] += loss.item()
                            epoch_loss += output.shape[0] * loss.item()
                            
                            if( i % 10 == 0 ):
                                correct_in_minibatch = ( local_labels == output.max(dim = 1)[1] ).sum()
                                print('[Net: %d, %d, %5d] loss: %.3f acc: %.3f' % (j + 1, epoch + 1, i + 1, (running_loss[j] / 10), correct_in_minibatch.item()/self.batch_size))
                                running_loss[j] = 0.0