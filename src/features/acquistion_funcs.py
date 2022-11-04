from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from sklearn.datasets import make_moons

import random

from src.models.train_model import train
from src.models.model import MLP
from src.features.utils import entropy_calc, bald_calc


def random_query(data_loader, query_size=10):
    
    sample_idx = []
    
    for X, y, idx in data_loader:
        sample_idx.extend(idx.tolist())
        
    random.shuffle(sample_idx)
    return sample_idx[0:query_size]


def margin_query(model, device, data_loader, query_size=10):
    
    margins = []
    indices = []
    
    model.eval()
    
    with torch.no_grad():
        for batch in data_loader:
        
            data, _, idx = batch
            logits = model(data.to(device))
            probabilities = F.softmax(logits, dim=1)
            
            # Select the top two class confidences for each sample
            toptwo = torch.topk(probabilities, 2, dim=1)[0]
            
            # Compute the margins = differences between the two top confidences
            differences = toptwo[:,0]-toptwo[:,1]
            margins.extend(torch.abs(differences).cpu().tolist())
            indices.extend(idx.tolist())

    margin = np.asarray(margins)
    index = np.asarray(indices)
    sorted_pool = np.argsort(margin)
    # Return the indices corresponding to the lowest `query_size` margins
    return index[sorted_pool][0:query_size]


def entropy_query(model, device, data_loader, query_size = 10):
    
    outputs = []
    indices = []
    
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            X, y, idx = batch
            outputs.append(F.softmax(model(X.to(device)), dim=1))
            indices.extend(idx.tolist())

    e = entropy_calc(outputs)

    ind = np.asarray(indices)
    sorted_pool = np.asarray(-e.detach().cpu()).argsort()
    
    return ind[sorted_pool][:query_size] 
     
def BALD_query(model, device, data_loader,
               query_size = 10, 
               T = 30, 
               method = 'MC_drop'):

    idxs = []
    outputs = []
    
    if method == 'MC_drop':
        model.train()
        with torch.no_grad():
            for t in range(T):
                outputs_inner = []
                for i, batch in enumerate(data_loader):
                    X, y, idx = batch
                    outputs_inner.append(F.softmax(model(X.to(device)), dim = 1))
                    idxs.extend(idx.tolist())
                outputs.append(torch.cat(outputs_inner, dim=0))
                             
    if method == 'ensemble':
        for t in range(T):
            outputs_inner = []
            optimizer = optim.Adam(model.parameters(), lr = 1e-4)
            model, optimizer = train(model, data_loader, optimizer, device, num_epochs = 50, plot = False, printout = False)
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    X, y, idx = batch
                    outputs_inner.append(F.softmax(model(X.to(device)), dim = 1))
                    idxs.extend(idx.tolist())
                outputs.append(torch.cat(outputs_inner, dim=0))
    
    bald_scores = bald_calc(outputs)
    #idxs = np.asarray(np.unique(idxs))    
    #sorted_pool = np.asarray(-bald_scores.detach().cpu()).argsort()

    idxs = np.asarray(np.unique(idxs))
    sorted_pool = np.asarray(-bald_scores.detach().cpu()).argsort()
    
    return idxs[sorted_pool][:query_size]

def query_the_oracle(model, 
                     poolloader, 
                     device,
                     T = 30,
                     query_size = 10, 
                     query_strategy = 'random', 
                     bald_method = 'MC_drop'):
    
    if query_strategy == 'random':
        sample_idx = random_query(poolloader, query_size=query_size)
        
    elif query_strategy == 'margin':
        sample_idx = margin_query(model, device, poolloader, query_size=query_size)
        
    elif query_strategy == 'entropy':
        sample_idx = entropy_query(model, device, poolloader, query_size=query_size)
        
    elif query_strategy == 'bald':
        sample_idx = BALD_query(model, device, poolloader, query_size = query_size, T = T, method = bald_method)
    
    return sample_idx