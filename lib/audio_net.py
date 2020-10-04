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

    def __init__(self, inputsize, outputsize, only_logsoftmax=False):
        super(AudioNet, self).__init__()
        self.only_logsoftmax = only_logsoftmax
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
        if( self.training or self.only_logsoftmax ):
            return self.log_softmax(x)
        else:
            return self.softmax(x)

class TinyAudioNet2(nn.Module):

    def __init__(self, inputsize, outputsize, only_logsoftmax=False):
        super(TinyAudioNet, self).__init__()
        self.only_logsoftmax = only_logsoftmax
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()
        self.dropOut = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(inputsize, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, outputsize)
		
    def forward(self, x):
        x = self.dropOut(self.relu( self.fc1(x) ))
        x = self.dropOut(self.relu( self.fc2(x) ))
        x = self.dropOut(self.relu( self.fc3(x) ))
        x = self.dropOut(self.relu( self.fc4(x) ))
        x = self.fc5(x)
        if( self.training or self.only_logsoftmax ):
            return self.log_softmax(x)
        else:
            return self.softmax(x)

class TinyAudioNet(nn.Module):

    def __init__(self, inputsize, outputsize, only_logsoftmax=False):
        super(TinyAudioNet, self).__init__()
        self.only_logsoftmax = only_logsoftmax
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(inputsize, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, outputsize)
		
    def forward(self, x):
        x = self.relu( self.fc1(x) )
        x = self.relu( self.fc2(x) )
        x = self.relu( self.fc3(x) )
        x = self.relu( self.fc4(x) )
        x = self.fc5(x)
        if( self.training or self.only_logsoftmax ):
            return self.log_softmax(x)
        else:
            return self.softmax(x)

class TinyAudioNetEnsemble(nn.Module):
    def __init__(self, modelA, modelB, modelC):
        super(TinyAudioNetEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        
    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x3 = self.modelC(x)
        return ( x1 + x2 + x3 ) / 3
            
class AudioNetTrainer:

    nets = []
    dataset_labels = []
    dataset_size = 0
    
    optimizers = []
    validation_loader = None
    train_loader = None
    criterion = nn.NLLLoss()
    batch_size = 256
    validation_split = .2
    max_epochs = 200
    random_seed = 42
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    def __init__(self, dataset, net_count = 1):
        self.net_count = net_count
        x, y = dataset[0]
        self.dataset_size = len(dataset)
        self.dataset_labels = dataset.get_labels()
        
        for i in range(self.net_count):
            self.nets.append(TinyAudioNet(len(x), len(self.dataset_labels), True))
            self.optimizers.append(optim.SGD(self.nets[i].parameters(), lr=0.003, momentum=0.9, nesterov=True))
 
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
        best_accuracy = []
        for i in range(self.net_count):
            self.nets[i] = self.nets[i].to(self.device)
            best_accuracy.append(0)
        starttime = int(time.time())
        
        with open(REPLAYS_FOLDER + "/model_training_" + filename + str(starttime) + ".csv", 'a', newline='') as csvfile:	
            headers = ['epoch', 'loss', 'avg_validation_accuracy']
            headers.extend(self.dataset_labels)
            writer = csv.DictWriter(csvfile, fieldnames=headers, delimiter=',')
            writer.writeheader()

            for epoch in range(self.max_epochs):
                # Training
                epoch_loss = 0.0
                running_loss = []                
                for j in range(self.net_count):
                    running_loss.append(0.0)
                    self.nets[j].train(True)
                i = 0
                with torch.set_grad_enabled(True):
                    for local_batch, local_labels in self.train_loader:
                        # Transfer to GPU
                        local_batch, local_labels = local_batch.to(self.device), local_labels.to(self.device)
                        
                        # Zero the gradients for this batch
                        i += 1                        
                        for j in range(self.net_count):
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
                    
                epoch_loss = epoch_loss / ( self.dataset_size * (1 - self.validation_split) )
                print('Training loss: {:.4f}'.format(epoch_loss))
                print( "Validating..." )
                for j in range(self.net_count):
                    self.nets[j].train(False)
                
                # Validation
                epoch_validation_loss = []
                correct = []
                epoch_loss = []
                accuracy = []                
                for j in range(self.net_count):
                    epoch_validation_loss.append(0.0)
                    correct.append(0)
                
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
                        for j in range(self.net_count):
                            optimizer = self.optimizers[j]
                            net = self.nets[j]
                            optimizer.zero_grad()
                            
                            # Calculating loss
                            output = net(local_batch)
                            correct[j] += ( local_labels == output.max(dim = 1)[1] ).sum().item()
                            loss = self.criterion(output, local_labels)
                            epoch_validation_loss[j] += output.shape[0] * loss.item()
                            
                            # Calculate the percentages
                            for index, label in enumerate(local_labels):
                                local_label_string = self.dataset_labels[label]
                                accuracy_batch['total'][local_label_string] += 1
                                if( output[index].argmax() == label ):
                                    accuracy_batch['correct'][local_label_string] += 1
                                accuracy_batch['percent'][local_label_string] = accuracy_batch['correct'][local_label_string] / accuracy_batch['total'][local_label_string]            

                for j in range(self.net_count):
                    epoch_loss.append(epoch_validation_loss[j] / ( self.dataset_size * self.validation_split ) )
                    accuracy.append( correct[j] / ( self.dataset_size * self.validation_split ) )
                    print('[Net: %d] Validation loss: %.4f accuracy %.3f' % (j + 1, epoch_loss[j], accuracy[j]))

                print('[Combined] Sum validation loss: %.4f average accuracy %.3f' % (np.sum(epoch_loss), np.average(accuracy)))
                csv_row = { 'epoch': epoch, 'loss': np.sum(epoch_loss), 'avg_validation_accuracy': np.average(accuracy) }
                for dataset_label in self.dataset_labels:
                    csv_row[dataset_label] = accuracy_batch['percent'][dataset_label]
                writer.writerow( csv_row )
                csvfile.flush()
                
                for j in range(self.net_count):                
                    current_filename = filename + '_' + str(j+1)
                    if( accuracy[j] > best_accuracy[j] ):
                        best_accuracy[j] = accuracy[j]
                        current_filename = filename + '_' + str(j+1) + '-BEST'
                        
                    torch.save({'state_dict': self.nets[j].state_dict(), 
                        'labels': self.dataset_labels,
                        'accuracy': accuracy[j],
                        'last_row': csv_row,
                        'loss': epoch_loss[j],
                        'epoch': epoch
                        }, os.path.join(CLASSIFIER_FOLDER, current_filename) + '-weights.pth.tar')    

        
