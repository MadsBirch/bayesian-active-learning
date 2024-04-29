import numpy as np
from sklearn.datasets import make_moons
from PIL import Image
import collections

import torch
import torchvision
import torch.utils.data as data
from torchvision import datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from typing import List

class ActiveLearningDataset_new():
    def __init__(self, X_train, Y_train, X_test, Y_test, handler):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler
        
        self.n_pool = len(X_train)
        self.n_test = len(X_test)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        
        
    def get_initial_pool(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
        
        
    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs])
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs])
    
    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train)
        
    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test)



class ActiveLearningDataset():
    
    """
    Takes a dataset and splits it into an active and available pool
    and appends a set of useful functions for updating these pools
    during the execution of the AL framework.
    """
    dataset: data.Dataset
    labeled_dataset: data.Dataset
    unlabeled_dataset: data.Dataset
    labeled_mask: np.ndarray
    unlabeled_mask: np.ndarray
    
    def __init__(self, dataset: data.Dataset):
        super().__init__()
        
        self.dataset = dataset
        self.labeled_mask = np.full((len(dataset),), False)
        self.unlabeled_mask = np.full((len(dataset),), True)

        self.labeled_dataset = data.Subset(self.dataset, None)
        self.unlabeled_dataset = data.Subset(self.dataset, None)
        
        self._update_indices()
        
    def _update_indices(self):
        self.labeled_dataset.indices = np.nonzero(self.labeled_mask)[0]
        self.unlabeled_dataset.indices = np.nonzero(self.unlabeled_mask)[0]
        
    def get_dataset_indices(self, unlabeled_indices: List[int]) -> List[int]:
        indices = self.unlabeled_dataset.indices[unlabeled_indices]
        return indices
    
    def acquire_samples(self, unlabeled_indices):
        #indices = self.get_dataset_indices(unlabeled_indices)

        self.labeled_mask[unlabeled_indices] = True
        self.unlabeled_mask[unlabeled_indices] = False
        self._update_indices()
        
    def remove_from_pool(self, unlabeled_indices):
        indices = self.get_dataset_indices(unlabeled_indices)

        self.unlabeled_mask[indices] = False
        self._update_indices()
        
    def get_random_pool_indices(self, size):
        assert 0 <= size <= len(self.unlabeled_dataset)
        pool_indices = torch.randperm(len(self.unlabeled_dataset))[:size]
        return pool_indices
        
    def extract_dataset_from_pool(self, size):
        """Extract a dataset randomly from the available dataset and make those indices unavailable."""
        return self.extract_dataset_from_indices(self.get_random_pool_indices(size))

    def extract_dataset_from_indices(self, unlabeled_indices):
        """Extract a dataset from the available dataset and make those indices unavailable."""
        dataset_indices = self.get_dataset_indices(unlabeled_indices)

        self.remove_from_pool(unlabeled_indices)
        return data.Subset(self.dataset, dataset_indices)
    
    def get_initial_balanced_trainset(self, n_classes: int, n_per_class: int):
        """Updates the initial training set to a balanced set"""
        initial_idx = []
        for r in range(n_classes):
            initial_idx.extend(np.random.choice(np.where(self.dataset.targets ==r)[0], size=n_per_class, replace=False))
        
        self.acquire_samples(initial_idx) 



class TwoMoons(data.Dataset):
    def __init__(self, X, targets, return_idx = True):
        self.X, self.targets = torch.tensor(X, dtype = torch.float), torch.tensor(targets, dtype = torch.long)
        self.unlabeled_mask = np.ones(len(self.targets))
        self.return_idx = return_idx
            
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        if self.return_idx:        
            return self.X[idx,:], self.targets[idx], idx
        else:
            return self.X[idx,:], self.targets[idx]
    
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


def get_balanced_sample_indices(target_classes: List, num_classes, n_per_digit=2) -> List[int]:
    """Given `target_classes` randomly sample `n_per_digit` for each of the `num_classes` classes."""
    permed_indices = torch.randperm(len(target_classes))

    if n_per_digit == 0:
        return []

    num_samples_by_class = collections.defaultdict(int)
    initial_samples = []

    for i in range(len(permed_indices)):
        permed_index = int(permed_indices[i])
        index, target = permed_index, int(target_classes[permed_index])

        num_target_samples = num_samples_by_class[target]
        if num_target_samples == n_per_digit:
            continue

        initial_samples.append(index)
        num_samples_by_class[target] += 1

        if len(initial_samples) == num_classes * n_per_digit:
            break

    return initial_samples


def get_targets(dataset):
    """Get the targets of a dataset without any target transforms.

    This supports subsets and other derivative datasets."""

    if isinstance(dataset, data.Subset):
        targets = get_targets(dataset.dataset)
        return torch.as_tensor(targets)[dataset.indices]
    if isinstance(dataset, data.ConcatDataset):
        return torch.cat([get_targets(sub_dataset) for sub_dataset in dataset.datasets])

    return torch.as_tensor(dataset.targets)