import numpy as np
from sklearn.datasets import make_moons
from PIL import Image
import collections

import torch
import torchvision
import torch.utils.data as data
from torchvision import datasets
import torchvision.transforms as transforms

from typing import List


class ActiveLearningDataset():
    
    """
    Takes a dataset and splits it into active and available pool
    and appends a set of useful functions for updating these pools
    during the execution of the AL framework.
    """
    dataset: data.Dataset
    training_dataset: data.Dataset
    pool_dataset: data.Dataset
    training_mask: np.ndarray
    pool_mask: np.ndarray
    
    def __init__(self, dataset: data.Dataset):
        super().__init__()
        
        self.dataset = dataset
        self.training_mask = np.full((len(dataset),), False)
        self.pool_mask = np.full((len(dataset),), True)

        self.training_dataset = data.Subset(self.dataset, None)
        self.pool_dataset = data.Subset(self.dataset, None)
        
        self._update_indices()
        
    def _update_indices(self):
        self.training_dataset.indices = np.nonzero(self.training_mask)[0]
        self.pool_dataset.indices = np.nonzero(self.pool_mask)[0]
        
    def get_dataset_indices(self, pool_indices: List[int]) -> List[int]:
        indices = self.pool_dataset.indices[pool_indices]
        return indices
    
    def acquire_samples(self, pool_indices):
        #indices = self.get_dataset_indices(pool_indices)

        self.training_mask[pool_indices] = True
        self.pool_mask[pool_indices] = False
        self._update_indices()
        
    def remove_from_pool(self, pool_indices):
        indices = self.get_dataset_indices(pool_indices)

        self.pool_mask[indices] = False
        self._update_indices()
        
    def get_random_pool_indices(self, size):
        assert 0 <= size <= len(self.pool_dataset)
        pool_indices = torch.randperm(len(self.pool_dataset))[:size]
        return pool_indices
        
    def extract_dataset_from_pool(self, size):
        """Extract a dataset randomly from the available dataset and make those indices unavailable."""
        return self.extract_dataset_from_indices(self.get_random_pool_indices(size))

    def extract_dataset_from_indices(self, pool_indices):
        """Extract a dataset from the available dataset and make those indices unavailable."""
        dataset_indices = self.get_dataset_indices(pool_indices)

        self.remove_from_pool(pool_indices)
        return data.Subset(self.dataset, dataset_indices)
    
    def get_initial_balanced_trainset(self, n_per_class: int):
        """Updates the initial training set to a balanced set"""
        initial_idx = []
        for r in range(10):
            initial_idx.extend(np.random.choice(np.where(self.dataset.targets ==r)[0], size=2, replace=False))
        
        self.acquire_samples(initial_idx) 



class TwoMoons(data.Dataset):
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
    
    def no_subset(self):
        self.subset_mask = np.zeros((len(self.subset_mask)))
        
        
        


def get_dataloaders(traindata, testdata,
                    batch_size = 256
                    ):
        
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=False, num_workers=0)
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return trainloader, testloader
