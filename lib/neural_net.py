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
        self.softmax = nn.LogSoftmax(dim=1)
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
        return self.softmax(x)
		
        
class AudioDataset(Dataset):
    samples = []
    paths = []
    length = 0

    def __init__(self, basedir, paths):
        self.paths = paths
		
        for index,path in enumerate(paths):
            totalpath = os.path.join(basedir,path)
            for file in os.listdir(totalpath):
                if( file.endswith(".wav") ):
                    full_filename = os.path.join(totalpath, file)
                    self.samples.append([full_filename,index])

    def feature_engineering_cached(self, filename):
        cached_filename = filename + "mf";
        if( os.path.isfile( cached_filename ) == False ):
            data_row, frequency = feature_engineering(filename)
            np.savetxt( cached_filename, data_row )
        
        return np.loadtxt( cached_filename, dtype='float' )

                    
    def __len__(self):
        return len( self.samples )

    def __getitem__(self, idx):
        filename = self.samples[idx][0]
        data_row = self.feature_engineering_cached(filename)
        return torch.tensor(data_row).float(), self.samples[idx][1]
		
    def get_labels(self):
        return self.paths

dataset = AudioDataset('C:\\Users\\anonymous\\Desktop\\Parrot.PY\\data\\recordings\\30ms', ['silence'
,'click_alveolar', 'fricative_f', 'fricative_v', 'sibilant_s', 'sibilant_sh', 'sibilant_z', 'sibilant_zh'
,'vowel_aa', 'vowel_ah', 'vowel_ae', 'vowel_e', 'vowel_eu', 'vowel_ih', 'vowel_iy', 'vowel_y', 'vowel_u', 'vowel_ow', 'vowel_oh'
#,'nasal_n', 'nasal_m', 'nasal_ng'
])
x, y = dataset[0]
criterion = nn.NLLLoss()
net = AudioNet(len(x), len(dataset.get_labels()))
optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)

batch_size = 256
validation_split = .2
random_seed = 42

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
max_epochs = 200

# Split the dataset into validation and training data loaders
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

print( len( x ) )

# Loop over epochs
net = net.to(device)
for epoch in range(max_epochs):
    # Training
    epoch_loss = 0.0
    running_loss = 0.0
    i = 0
    net.train(True)
    with torch.set_grad_enabled(True):
        for local_batch, local_labels in train_loader:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            
            # Zero the gradients for this batch
            optimizer.zero_grad()
            
            # Calculating loss
            output = net(local_batch)
            loss = criterion(output, local_labels)
            loss.backward()
            
            # Prevent exploding weights
            torch.nn.utils.clip_grad_norm_(net.parameters(),4)
            optimizer.step()
            
            running_loss += loss.item()
            epoch_loss += output.shape[0] * loss.item()     
            i += 1
            if( i % 10 == 0 ):
                correct_in_minibatch = ( local_labels == output.max(dim = 1)[1] ).sum()
                print('[%d, %5d] loss: %.3f acc: %.3f' % (epoch + 1, i + 1, (running_loss / 10), correct_in_minibatch.item()/batch_size))
                running_loss = 0.0
        
    epoch_loss = epoch_loss / ( len(dataset) * (1 - validation_split) )
    print('Training loss: {:.4f}'.format(epoch_loss))
    print( "Validating..." )
    # Validation
    net.train(False)
    epoch_validation_loss = 0.0
    correct = 0
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_loader:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
			
            # Zero the gradients for this batch
            optimizer.zero_grad()
        
		    # Calculating loss
            output = net(local_batch)
            correct += ( local_labels == output.max(dim = 1)[1] ).sum().item()
            loss = criterion(output, local_labels)
            epoch_validation_loss += output.shape[0] * loss.item()
			
    epoch_loss = epoch_validation_loss / ( len(dataset) * validation_split )
    accuracy = correct / ( len(dataset) * validation_split )
    print('Validation loss: {:.4f} accuracy {:.3f}'.format(epoch_loss, accuracy))
    torch.save(net.state_dict(), 'data//models//torch-8884-exnasals.pth.tar')