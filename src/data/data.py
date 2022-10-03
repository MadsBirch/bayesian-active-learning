import numpy as np
from sklearn.datasets import make_moons

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets


class TwoMoons(Dataset):
    def __init__(self, moons_data):
        X, y = moons_data
        self.X, self.y = torch.tensor(X, dtype = torch.float), torch.tensor(y, dtype = torch.long)
        self.unlabeled_mask = np.ones(len(self.y))
            
    def __len__(self):

        return len(self.y)
    
    def __getitem__(self, idx):        
        return self.X[idx,:], self.y[idx], idx
    
    def update_mask(self, idx):
        self.unlabeled_mask[idx] = 0
        
    def reset_mask(self): 
        self.unlabeled_mask = np.ones((len(self.unlabeled_mask)))
    
   
class CIFAR10_CUSTOM(datasets.CIFAR10):
    def __init__(self, root, train=True):
        super().__init__(root, train, download=True)
        self.unlabeled_mask = np.ones(len(super().__len__()))
        
    def update_mask(self, idx):
        self.unlabeled_mask[idx] = 0
        
    def reset_mask(self): 
        self.unlabeled_mask = np.ones((len(self.unlabeled_mask)))
        
    
def get_dataloaders(moons_data, 
                    batch_size = 256,
                    val_split = 0.2,
                    ):
    # load in data
    data = TwoMoons(moons_data)
    
    # split into train and validation
    data_lengths = [int(len(data)*(1-val_split)), int(len(data)*val_split)]
    traindata, testdata = random_split(data, data_lengths)
    
    # split validation into val and test
    #val_lengths= [int(len(valdata)*(1-val_split)), int(len(valdata)*(val_split))]
    #valdata, testdata = random_split(valdata, val_lengths)
    
    
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=0)
    #testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return trainloader, testloader
