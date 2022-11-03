import numpy as np
from sklearn.datasets import make_moons
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as transforms

class TwoMoons(Dataset):
    def __init__(self, X, y, return_idx = True):
        self.X, self.y = torch.tensor(X, dtype = torch.float), torch.tensor(y, dtype = torch.long)
        self.unlabeled_mask = np.ones(len(self.y))
        self.return_idx = return_idx
            
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        if self.return_idx:        
            return self.X[idx,:], self.y[idx], idx
        else:
            return self.X[idx,:], self.y[idx]
    
    def update_mask(self, idx):
        self.unlabeled_mask[idx] = 0
        
    def reset_mask(self): 
        self.unlabeled_mask = np.ones((len(self.unlabeled_mask)))
    
   
class MNIST_CUSTOM(datasets.MNIST):
    def __init__(self, root, transform = transforms.ToTensor(), train=True):
        super().__init__(root, train, download=True, transform=transform)
        self.unlabeled_mask = np.ones(super().__len__())
        self.subset_mask = np.ones(super().__len__())
        
    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, idx
        
    def update_mask(self, idx):
        self.unlabeled_mask[idx] = 0
        
    def reset_mask(self): 
        self.unlabeled_mask = np.ones((len(self.unlabeled_mask)))
    
    def update_submask(self, idx):
        self.subset_mask[idx] = 0
        
    def reset_submask(self): 
        self.subset_mask = np.ones((len(self.subset_mask)))
        
        
        


def get_dataloaders(traindata, testdata,
                    batch_size = 256
                    ):
        
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=False, num_workers=0)
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return trainloader, testloader
