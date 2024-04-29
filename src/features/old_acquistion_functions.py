import random
import numpy as np
from typing import List
from dataclasses import dataclass
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

from src.models.train_model import train
#import src.models.joint_entropy as joint_entropy

@dataclass
class QueryBatch():
    indices: List[int]
    scores: List[float]
    
    
def compute_entropy(outputs):
    outputs = torch.cat(outputs, dim = 0)
    H = (-outputs*torch.log(outputs + 1e-10)).sum(1)
    
    return H


def compute_bald(outputs):
    outputs = torch.stack(outputs, dim = -1)
    pc = outputs.mean(2)
    
    H = (-pc*torch.log(pc + 1e-10)).sum(1)
    E_H = -(outputs*torch.log(outputs + 1e-10)).sum(1).mean(1)
    bald = H - E_H
    
    return bald
    

def random_query(dataloader, 
                 query_size: int
                 ) -> QueryBatch:
    
    sample_idx = []
    
    for X, y, idx in dataloader:
        sample_idx.extend(idx.tolist())
        
    random.shuffle(sample_idx)
    
    idxs = sample_idx[:query_size]
    scores = np.zeros(query_size)
    
    return QueryBatch(idxs, scores)


def margin_query(device, model: nn.Module, 
                 dataloader: data.DataLoader, 
                 query_size: int
                 ) -> QueryBatch:
    
    margins = []
    indices = []
    
    model.eval()
    
    with torch.no_grad():
        for X, y, idx in dataloader:
    
            logits = model(X.to(device))
            p_hat = F.softmax(logits, dim=1)
            
            # Select the top two class confidences for each sample
            toptwo = torch.topk(p_hat, 2, dim=1)[0]
            
            # Compute the margins = differences between the two top confidences
            differences = toptwo[:,0]-toptwo[:,1]
            margins.extend(torch.abs(differences).cpu().tolist())
            indices.extend(idx.tolist())

    margin = np.asarray(margins)
    index = np.asarray(indices)
    sorted_pool = np.argsort(margin)
    
    idxs = index[sorted_pool][:query_size]
    scores = sorted_pool[:query_size]
    
    return QueryBatch(idxs, scores) 

def entropy_query(device,
                  model: nn.Module,  
                  dataloader: data.DataLoader, 
                  query_size: int
                  ) -> QueryBatch:
    
    outputs = []
    indices = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            X, y, idx = batch
            outputs.append(F.softmax(model(X.to(device)), dim=1))
            indices.extend(idx.tolist())

    scores = compute_entropy(outputs)

    ind = np.asarray(indices)
    sorted_pool = np.asarray(-scores.detach().cpu()).argsort()
    
    idxs = ind[sorted_pool][:query_size] 
    scores = sorted_pool[:query_size]
    
    return QueryBatch(idxs, scores)

     
def BALD_query(device,
               model: nn.Module,  
               dataloader: data.DataLoader,
               query_size: int, 
               T: int, 
               method: str
               ) -> QueryBatch:

    idxs = []
    outputs = []
    
    ### CALCULATE LOGITS BOX ###
    if method == 'MC_drop':
        model.train()
        with torch.no_grad():
            for t in range(T):
                outputs_inner = []
                for i, batch in enumerate(dataloader):
                    X, y, idx = batch
                    outputs_inner.append(F.softmax(model(X.to(device)), dim = 1))
                    idxs.extend(idx.tolist())
                outputs.append(torch.cat(outputs_inner, dim=0))
                             
    if method == 'ensemble':
        for t in range(T):
            outputs_inner = []
            optimizer = optim.Adam(model.parameters(), lr = 1e-3)
            model, optimizer = train(model, dataloader, optimizer, device, num_epochs = 50, plot = False, printout = False)
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    X, y, idx = batch
                    outputs_inner.append(F.softmax(model(X.to(device)), dim = 1))
                    idxs.extend(idx.tolist())
                outputs.append(torch.cat(outputs_inner, dim=0))
    
    scores = compute_bald(outputs)

    indices = np.asarray(np.unique(idxs))
    sorted_pool = np.asarray(-scores.detach().cpu()).argsort()
    
    idxs = indices[sorted_pool][:query_size]
    
    # these scores are not correct
    scores = sorted_pool[:query_size]
    
    return QueryBatch(idxs, scores)


    
def query_the_oracle(device,
                     model: nn.Module, 
                     unlabeled_set: data.Dataset, 
                     T: int,
                     query_size: int, 
                     query_strategy: str, 
                     bald_method: str):
    
    poolloader = data.DataLoader(unlabeled_set, batch_size = 256, shuffle=False, num_workers=0)
    
    if query_strategy == 'random':
        query_batch = random_query(poolloader, query_size=query_size)
        
    elif query_strategy == 'margin':
        query_batch = margin_query(device, model, poolloader, query_size=query_size)
        
    elif query_strategy == 'entropy':
        query_batch = entropy_query(device, model, poolloader, query_size=query_size)
        
    elif query_strategy == 'bald':
        query_batch = BALD_query(device, model, poolloader, query_size = query_size, T = T, method = bald_method)
    
    return query_batch


##------FOR LATER---------## 
""""

def compute_conditional_entropy(probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Conditional Entropy", leave=False)

    def compute(probs_n_K_C, start: int, end: int):
        nats_n_K_C = probs_n_K_C * torch.log(probs_n_K_C)
        nats_n_K_C[probs_n_K_C == 0] = 0.0

        entropies_N[start:end].copy_(-torch.sum(nats_n_K_C, dim=(1, 2)) / K)
        pbar.update(end - start)

    pbar.close()

    return entropies_N


def compute_entropy(probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Entropy", leave=False)

    def compute(probs_n_K_C, start: int, end: int):
        mean_probs_n_C = probs_n_K_C.mean(dim=1)
        nats_n_C = mean_probs_n_C * torch.log(mean_probs_n_C)
        nats_n_C[mean_probs_n_C == 0] = 0.0

        entropies_N[start:end].copy_(-torch.sum(nats_n_C, dim=1))
        pbar.update(end - start)

    pbar.close()

    return entropies_N


def batchBALD_query(log_probs_N_K_C: torch.Tensor, 
                    batch_size: int, 
                    num_samples: int, 
                    dtype=None, 
                    device=None 
                    ) -> QueryBatch:
    
    N, K, C = log_probs_N_K_C.shape

    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    if batch_size == 0:
        return QueryBatch(candidate_scores, candidate_indices)

    conditional_entropies_N = compute_conditional_entropy(log_probs_N_K_C)

    batch_joint_entropy = joint_entropy.DynamicJointEntropy(
        num_samples, batch_size - 1, K, C, dtype=dtype, device=device
    )

    # We always keep these on the CPU.
    scores_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())

    for i in tqdm(range(batch_size), desc="BatchBALD", leave=False):
        if i > 0:
            latest_index = candidate_indices[-1]
            batch_joint_entropy.add_variables(log_probs_N_K_C[latest_index : latest_index + 1])

        shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

        batch_joint_entropy.compute_batch(log_probs_N_K_C, output_entropies_B=scores_N)

        scores_N -= conditional_entropies_N + shared_conditinal_entropies
        scores_N[candidate_indices] = -float("inf")

        candidate_score, candidate_index = scores_N.max(dim=0)

        candidate_indices.append(candidate_index.item())
        candidate_scores.append(candidate_score.item())

    return QueryBatch(candidate_scores, candidate_indices)


class AcquisitionFunction():
    def __init__(self,
                device,
                model: nn.Module, 
                poolloader: data.DataLoader, 
                T: int,
                query_size: int, 
                query_strategy: str, 
                bald_method: str) -> None:
        
        self.device = device
        self.model = model
        self.poolloader = poolloader 
        self.T = T
        self.query_size = query_size 
        self.query_strategy = query_strategy
        self.bald_method = bald_method
    
    def get_samples(self):
        pass
        
class BALD():
    def __init__(self):
        pass
        
    def get_samples(self):
        pass
    
    def get_grid(self):
        pass
        
"""