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
from lib.typing import RunSettings
from typing import Optional
import random

class TinyAudioNet(nn.Module):

    def __init__(self, inputsize, outputsize, only_logsoftmax=False):
        super(TinyAudioNet, self).__init__()
        self.only_logsoftmax = only_logsoftmax
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.selu = nn.SELU()
        self.dropOut = nn.Dropout(p=0.15)
        
        self.batchNorm = nn.BatchNorm1d(inputsize)        
        self.fc1 = nn.Linear(inputsize, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, outputsize)
		
    def forward(self, x):
        x = self.dropOut(self.selu( self.fc1(self.batchNorm(x))))
        x = self.dropOut(self.selu( self.fc2(x) ))
        x = self.dropOut(self.selu( self.fc3(x) ))
        x = self.dropOut(self.selu( self.fc4(x) ))
        x = self.dropOut(self.selu( self.fc5(x) ))
        x = self.fc6(x)
        if( self.training or self.only_logsoftmax ):
            return self.log_softmax(x)
        else:
            return self.softmax(x)

class TinyAudioNetEnsemble(nn.Module):
    def __init__(self, models):
        super(TinyAudioNetEnsemble, self).__init__()
        self.models = []
        self.model_length = len(models)
        for model in models:
            #model.double()
            self.models.append(model)
            
    def forward(self, x):
        out = 0
        for index, model in enumerate(self.models):
            if (index == 0):
                out = model(x)
            else:
                out = out + model(x)
        
        return out / self.model_length
            
class AudioNetTrainer:
    dataset_size = 0
    
    criterion = nn.NLLLoss()
    batch_size = 512
    validation_split = .2
    max_epochs = 300
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dataset = False
    input_size = 120
    
    def __init__(self, dataset, net_count = 1, audio_settings = None, run_settings: Optional[RunSettings] = None):
        self.nets = []
        self.optimizers = []
        self.random_seeds = []
        self.train_indices = []
        self.train_loaders = []
        self.validation_loaders = []

        self.net_count = net_count
        x, y = dataset[0]
        self.input_size = len(x)
        self.dataset_labels = dataset.get_labels()
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.audio_settings = audio_settings
        self.run_settings = run_settings or {}
        self.dataset_size = len(dataset)
        
        split = int(np.floor(self.validation_split * self.dataset_size))

        for i in range(self.net_count):
            self.nets.append(TinyAudioNet(self.input_size, len(self.dataset_labels), True))
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
        
    def train(self, filename: str, batch_callback=None, epoch_callback=None, stop_check=None):
        """Train every net, saving the best combined model as it improves.

        Args:
            filename: what the model is saved as.
            batch_callback: called every 10 batches with
                (net, epoch, batch, loss, accuracy).
            epoch_callback: called after each epoch with
                (epoch, loss, net_accuracies, label_accuracy, new_best).
            stop_check: called at those same points. Return True to stop.
        """
        best_accuracy = []
        combined_classifier_map = {}
        for i in range(self.net_count):
            self.nets[i] = self.nets[i].to(self.device)
            combined_classifier_map['classifier_' + str(i)] = os.path.join(CLASSIFIER_FOLDER, filename + '_' + str(i + 1) + '-BEST-weights.pth.tar')
            best_accuracy.append(0)
        starttime = int(time.time())
        combined_model = TinyAudioNetEnsemble(self.nets).to(self.device)
        
        input_size = 120

        label_frames = {label: 0 for label in self.dataset_labels}
        for sample in self.dataset.samples:
            label_frames[self.dataset_labels[sample[1]]] += 1
        
        os.makedirs(REPLAYS_FOLDER, exist_ok=True)
        with open(REPLAYS_FOLDER + "/model_training_" + filename + str(starttime) + ".csv", 'a', newline='') as csvfile:	
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
                        st_batch= time.time()
                        for local_batch, local_labels in self.train_loaders[j]:
                            # PyTorch needs 2+ rows per batch while training.
                            if local_batch.size(0) < 2:
                                continue

                            # Transfer to GPU
                            local_batch, local_labels = local_batch.to(self.device), local_labels.to(self.device)
                            
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
                                print('[Net: %d, %d, %5d] loss: %.3f acc: %.3f' % (j + 1, epoch + 1, i + 1, (running_loss[j] / 10), correct_in_minibatch.item()/local_labels.size(0)))
                                if batch_callback is not None:
                                    batch_callback(j + 1, epoch, i, running_loss[j] / 10, correct_in_minibatch.item()/local_labels.size(0))
                                running_loss[j] = 0.0
                                if stop_check is not None and stop_check():
                                    print("Stop requested - Stopped training loop")
                                    print( "------------------------------------------------------")
                                    return
                    
                epoch_loss = epoch_loss / ( self.dataset_size * (1 - self.validation_split) )
                print('Training loss: {:.4f}'.format(epoch_loss))
                print( "Validating..." )
                for j in range(self.net_count):
                    self.nets[j].train(False)
                
                # Validation
                self.dataset.set_training(False)
                epoch_validation_loss = []
                correct = []
                epoch_loss = []
                accuracy = []
                combined_correct = 0
                label_accuracy = []
                for j in range(self.net_count):
                    epoch_validation_loss.append(0.0)
                    correct.append(0)
                
                    with torch.set_grad_enabled(False):
                        accuracy_batch = {'total': {}, 'correct': {}, 'percent': {}}
                        for dataset_label in self.dataset_labels:
                            accuracy_batch['total'][dataset_label] = 0
                            accuracy_batch['correct'][dataset_label] = 0
                            accuracy_batch['percent'][dataset_label] = 0
                    
                        for local_batch, local_labels in self.validation_loaders[j]:
                            # Transfer to GPU
                            local_batch, local_labels = local_batch.to(self.device), local_labels.to(self.device)
                            
                            # Zero the gradients for this batch
                            optimizer = self.optimizers[j]
                            net = self.nets[j]
                            optimizer.zero_grad()
                            
                            # Calculating loss
                            output = net(local_batch)
                            correct[j] += ( local_labels == output.max(dim = 1)[1] ).sum().item()
                            loss = self.criterion(output, local_labels)
                            epoch_validation_loss[j] += output.shape[0] * loss.item()
                            
                            # Calculate combined accuracy on last validation pass
                            if (j + 1 == self.net_count):
                                combined_output = combined_model(local_batch)
                                combined_correct += ( local_labels == combined_output.max(dim = 1)[1] ).sum().item()
                            
                            # Calculate the percentages
                            for index, label in enumerate(local_labels):
                                local_label_string = self.dataset_labels[label]
                                accuracy_batch['total'][local_label_string] += 1
                                if( output[index].argmax() == label ):
                                    accuracy_batch['correct'][local_label_string] += 1
                                accuracy_batch['percent'][local_label_string] = accuracy_batch['correct'][local_label_string] / accuracy_batch['total'][local_label_string]
                
                        label_accuracy.append(accuracy_batch['percent'])

                mean_label_accuracy = {}
                for dataset_label in self.dataset_labels:
                    scores = [p[dataset_label] for p in label_accuracy]
                    mean_label_accuracy[dataset_label] = sum(scores) / len(scores)
                
                for j in range(self.net_count):
                    epoch_loss.append(epoch_validation_loss[j] / ( self.dataset_size * self.validation_split ) )
                    accuracy.append( correct[j] / ( self.dataset_size * self.validation_split ) )
                    print('[Net: %d] Validation loss: %.4f accuracy %.3f' % (j + 1, epoch_loss[j], accuracy[j]))

                combined_accuracy = combined_correct / ( self.dataset_size * self.validation_split )
                print('[Combined] Sum validation loss: %.4f average accuracy %.3f' % (np.sum(epoch_loss), combined_accuracy))
                
                csv_row = { 'epoch': epoch, 'loss': np.sum(epoch_loss), 'avg_validation_accuracy': np.average(accuracy) }
                for dataset_label in self.dataset_labels:
                    csv_row[dataset_label] = mean_label_accuracy[dataset_label]
                writer.writerow( csv_row )
                csvfile.flush()
                                
                new_best = False
                for j in range(self.net_count):
                    current_filename = filename + '_' + str(j+1)
                    if( accuracy[j] > best_accuracy[j] ):
                        best_accuracy[j] = accuracy[j]
                        current_filename = filename + '_' + str(j+1) + '-BEST'
                        new_best = True
                        
                    torch.save({'state_dict': self.nets[j].state_dict(), 
                        'input_size': self.input_size,
                        'labels': self.dataset_labels,
                        'accuracy': accuracy[j],
                        'last_row': csv_row,
                        'loss': epoch_loss[j],
                        'epoch': epoch,
                        'random_seed': self.random_seeds[j],
                        'label_accuracy': label_accuracy[j],
                        'combined_accuracy': combined_accuracy,
                        'label_frames': label_frames,
                        'trained_at': starttime,
                        'run_settings': self.run_settings,
                        }, os.path.join(CLASSIFIER_FOLDER, current_filename) + '-weights.pth.tar')
                
                # Persist a new combined model with the best weights if new best weights are given
                if (new_best == True):
                    print( "------------------------------------------------------")
                    print( "Persisting new combined best in " + filename )
                    print( "------------------------------------------------------")                    
                    connect_model( filename, combined_classifier_map, "ensemble_torch", True, self.audio_settings )
                
                if epoch_callback is not None:
                    epoch_callback(epoch, np.sum(epoch_loss), accuracy, mean_label_accuracy, new_best)
                if stop_check is not None and stop_check():
                    print("Stop requested - Stopped training loop")
                    print( "------------------------------------------------------")
                    return

                with KeyPoller() as key_poller:
                    ESCAPEKEY = '\x1b'
                    character = key_poller.poll()
                    if ( character == ESCAPEKEY ):
                        print("Pressed escape - Stopped training loop")
                        print( "------------------------------------------------------")
                        return