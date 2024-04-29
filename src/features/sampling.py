from typing import List
from dataclasses import dataclass
import random
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.train_model import train

### helper functions ###
@dataclass
class QueryBatch():
    indices: List[int]
    scores: List[float]

def compute_entropy(probs):
    
    # Adding a small value to probabilities to avoid log(0).
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
    return entropy

def compute_bald(outputs):
    # Assuming outputs shape is [num_samples, num_classes, n_drop]
    pc = outputs.mean(dim=2)
    H = -1 * (pc * torch.log(pc + 1e-9)).sum(dim=1)  # Entropy of the average prediction

    # Should compute as negative entropy first, then take the mean
    E_H = -1 * (outputs * torch.log(outputs + 1e-9)).sum(dim=1).mean(dim=1)  # Average negative entropy

    bald = H - E_H  # Correctly subtracting the expected entropy from the entropy of the mean prediction
    
    return bald

### sampling class ###
class Sampling():
    def __init__(self, data: data.Dataset, model: nn.Module, query_size: int, device: str):
        self.query_size = query_size
        self.data = data
        self.model = model
        self.device = device
    
    def random(self):
        
        scores = np.zeros(self.query_size)     
        indices = np.random.choice(self.data.indices, self.query_size, replace=False)
                
        return QueryBatch(indices, scores)

    
    def margin(self):
        unlabeled_idxs = self.data.indices
        probs = self.model.predict_probs(self.data)
        
        # Sort probabilities in descending order.
        probs_sorted, _ = probs.sort(descending=True)
        
        # Compute margins as the difference between the top two class probabilities.
        margins = probs_sorted[:, 0] - probs_sorted[:, 1]
        
        # Sort margins to get the smallest margins first (indicative of highest uncertainty).
        # We do not need to sort in descending order here since we're interested in the smallest values.
        sorted_indices = margins.sort()[1][:self.query_size].cpu()
        
        indices = unlabeled_idxs[sorted_indices]
        scores = margins[sorted_indices]
        
        return QueryBatch(indices, scores)
    
    def entropy(self):
        unlabeled_idxs = self.data.indices
        probs = self.model.predict_probs(self.data)
        
        H = compute_entropy(probs)
        
        # Sort indices based on entropy in descending order using PyTorch functionalities.
        sorted_indices_desc = torch.argsort(H, descending=True)
        
        # Select the top `query_size` indices.
        top_indices = sorted_indices_desc[:self.query_size].cpu()
        indices = unlabeled_idxs[top_indices]
        scores = H[top_indices]
        
        return QueryBatch(indices, scores)
    
    
    def bald_mc(self, n_drop: int):
        unlabeled_idxs = self.data.indices
        probs = self.model.predict_probs_mc(self.data, n_drop=n_drop)
        bald_scores = compute_bald(probs)

        sorted_indices = torch.argsort(bald_scores, descending=True)[:self.query_size].cpu()
        indices = unlabeled_idxs[sorted_indices]
        scores = bald_scores[sorted_indices]

        return QueryBatch(indices, scores)
    
    
    def bald_ensemble(self, traindata, n_ens = 5, n_epochs = 50, lr = 1e-3):
               
        unlabeled_idxs = self.data.indices
        probs = self.model.predict_probs_ensemble(self.data, traindata = traindata, n_ens = n_ens, n_epochs = n_epochs, lr = lr)
        bald_scores = compute_bald(probs)

        sorted_indices = torch.argsort(bald_scores, descending=True)[:self.query_size].cpu()
        indices = unlabeled_idxs[sorted_indices]
        scores = bald_scores[sorted_indices]

        return QueryBatch(indices, scores)
        
    
    def batch_bald_sampling():
        return QueryBatch