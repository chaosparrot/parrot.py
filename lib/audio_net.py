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
            
            
class AudioNetTrainer:

    net = None
    dataset_labels = []
    dataset_size = 0
    
    optimizer = None
    validation_loader = None
    train_loader = None
    criterion = nn.NLLLoss()
    batch_size = 256
    validation_split = .2
    max_epochs = 200
    random_seed = 42
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    def __init__(self, dataset):
        x, y = dataset[0]
        self.dataset_size = len(dataset)
        self.dataset_labels = dataset.get_labels()
        self.net = AudioNet(len(x), len(self.dataset_labels))
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.003, momentum=0.9)

        # Split the dataset into validation and training data loaders
        indices = list(range(self.dataset_size))
        split = int(np.floor(self.validation_split * self.dataset_size))
        np.random.seed(self.random_seed)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler)
        self.validation_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, sampler=valid_sampler)
        
        
    def train(self, filename):
        self.net = self.net.to(self.device)
        starttime = int(time.time())
        
        best_accuracy = 0
        
        with open(REPLAYS_FOLDER + "/model_training_" + filename + str(starttime) + ".csv", 'a', newline='') as csvfile:	
            headers = ['epoch', 'loss', 'validation_accuracy']
            headers.extend(self.dataset_labels)
            writer = csv.DictWriter(csvfile, fieldnames=headers, delimiter=',')
            writer.writeheader()

            for epoch in range(self.max_epochs):
                # Training
                epoch_loss = 0.0
                running_loss = 0.0
                i = 0
                self.net.train(True)
                with torch.set_grad_enabled(True):
                    for local_batch, local_labels in self.train_loader:
                        # Transfer to GPU
                        local_batch, local_labels = local_batch.to(self.device), local_labels.to(self.device)
                        
                        # Zero the gradients for this batch
                        self.optimizer.zero_grad()
                        
                        # Calculating loss
                        output = self.net(local_batch)
                        loss = self.criterion(output, local_labels)
                        loss.backward()
                                    
                        # Prevent exploding weights
                        torch.nn.utils.clip_grad_norm_(self.net.parameters(),4)
                        self.optimizer.step()
                        
                        running_loss += loss.item()
                        epoch_loss += output.shape[0] * loss.item()     
                        i += 1
                        if( i % 10 == 0 ):
                            correct_in_minibatch = ( local_labels == output.max(dim = 1)[1] ).sum()
                            print('[%d, %5d] loss: %.3f acc: %.3f' % (epoch + 1, i + 1, (running_loss / 10), correct_in_minibatch.item()/self.batch_size))
                            running_loss = 0.0
                    
                epoch_loss = epoch_loss / ( self.dataset_size * (1 - self.validation_split) )
                print('Training loss: {:.4f}'.format(epoch_loss))
                print( "Validating..." )
                
                # Validation
                self.net.train(False)
                epoch_validation_loss = 0.0
                correct = 0
                with torch.set_grad_enabled(False):
                    accuracy_batch = {'total': {}, 'correct': {}, 'percent': {}}
                    for dataset_label in self.dataset_labels:
                        accuracy_batch['total'][dataset_label] = 0
                        accuracy_batch['correct'][dataset_label] = 0
                        accuracy_batch['percent'][dataset_label] = 0            
                
                    for local_batch, local_labels in self.validation_loader:
                        # Transfer to GPU
                        local_batch, local_labels = local_batch.to(self.device), local_labels.to(self.device)
                        
                        # Zero the gradients for this batch
                        self.optimizer.zero_grad()
                            
                        # Calculating loss
                        output = self.net(local_batch)
                        correct += ( local_labels == output.max(dim = 1)[1] ).sum().item()
                        loss = self.criterion(output, local_labels)
                        epoch_validation_loss += output.shape[0] * loss.item()
                        
                        # Calculate the percentages
                        for index, label in enumerate(local_labels):
                            local_label_string = self.dataset_labels[label]
                            accuracy_batch['total'][local_label_string] += 1
                            if( output[index].argmax() == label ):
                                accuracy_batch['correct'][local_label_string] += 1
                            accuracy_batch['percent'][local_label_string] = accuracy_batch['correct'][local_label_string] / accuracy_batch['total'][local_label_string]            
                     

                epoch_loss = epoch_validation_loss / ( self.dataset_size * self.validation_split )
                accuracy = correct / ( self.dataset_size * self.validation_split )
                print('Validation loss: {:.4f} accuracy {:.3f}'.format(epoch_loss, accuracy))
                
                csv_row = { 'epoch': epoch, 'loss': epoch_loss, 'validation_accuracy': accuracy }
                for dataset_label in self.dataset_labels:
                    csv_row[dataset_label] = accuracy_batch['percent'][dataset_label]
                writer.writerow( csv_row )
                csvfile.flush()
                
                current_filename = filename
                if( accuracy > best_accuracy ):
                    best_accuracy = accuracy
                    current_filename = filename + '-BEST'
                    
                torch.save({'state_dict': self.net.state_dict(), 
                    }, os.path.join(CLASSIFIER_FOLDER, current_filename) + '-weights.pth.tar')    

        
