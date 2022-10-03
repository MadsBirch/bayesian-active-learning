from random import shuffle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.datasets import make_moons

def random_query(data_loader, query_size=10):
    
    sample_idx = []
    
    # Because the data has already been shuffled inside the data loader,
    # we can simply return the `query_size` first samples from it
    for X, y, idx in data_loader:
        
        sample_idx.extend(idx.tolist())

        if len(sample_idx) >= query_size:
            break
        
        
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


def least_confidence_query(model, device, data_loader, query_size=10):

    confidences = []
    indices = []
    
    model.eval()
    
    with torch.no_grad():
        for batch in data_loader:
        
            data, _, idx = batch
            logits = model(data.to(device))
            probabilities = F.softmax(logits, dim=1)
            
            # Keep only the top class confidence for each sample
            most_probable = torch.max(probabilities, dim=1)[0]
            confidences.extend(most_probable.cpu().tolist())
            indices.extend(idx.tolist())
            
    conf = np.asarray(confidences)
    ind = np.asarray(indices)
    sorted_pool = np.argsort(conf)
    # Return the indices corresponding to the lowest `query_size` confidences
    return ind[sorted_pool][0:query_size]

def entropy_query(model, device, data_loader, query_size = 10):
    
    e_scores = []
    indices = []
    
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            data, _, idx = batch
            logits = model(data.to(device))
            
            e = -1.0 * torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), dim=1)
            e_scores.extend(e.cpu().tolist())
            indices.extend(idx.tolist())

    conf = np.asarray(e_scores)
    ind = np.asarray(indices)
    sorted_pool = np.argsort(conf)
    
    return ind[sorted_pool][-query_size:] 
    #return indices, e_scores
    
    
def BALD_query(model, device, data_loader, batch_size, query_size = 10, T = 30, method = 'MC_drop'):

    # get appox
    model.train()
    
    indices = []
    if method == 'MC_drop':
        logits = torch.zeros((len(data_loader)*batch_size,2,T))
        for t in range(T):
            for i, batch in enumerate(data_loader):
                X, y, idx = batch
                logit = model(X.to(device))
                logits[i*len(y):i*len(y)+len(y),:,t] = logit
                
                indices.extend(idx.tolist())


        first_term = -1.0 * torch.sum(F.softmax((1/T)*torch.sum(logits, dim = 2), dim=1) * F.log_softmax((1/T)*torch.sum(logits, dim = 2), dim=1), dim=1)
        second_term = (1/T)*torch.sum(torch.sum(F.softmax(logits, dim = 1)*F.log_softmax(logits, dim=1), dim = 2),dim = 1)

        BALD_scores = first_term+second_term
            
    if method == 'Laplace':
        print('Not implemented yet!')
        
        
    conf = np.asarray(BALD_scores.cpu().tolist())
    ind = np.asarray(indices)
    sorted_pool = np.argsort(conf)
        
    return ind[sorted_pool][0:query_size]

def query_the_oracle(model, dataset, device, T = 30, query_size = 10, query_strategy = 'random', batch_size = 256):
    
    unlabeled_idx = np.nonzero(dataset.unlabeled_mask)[0]
    
    pool_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, 
                                sampler=SubsetRandomSampler(unlabeled_idx), shuffle=False)
    
    if query_strategy == 'random':
        sample_idx = random_query(pool_loader, query_size=query_size)
        
    elif query_strategy == 'margin':
        sample_idx = margin_query(model, device, pool_loader, query_size=query_size)
        
    elif query_strategy == 'least_conf':
        sample_idx = least_confidence_query(model, device, pool_loader, query_size=query_size)
        
    elif query_strategy == 'entropy':
        sample_idx = entropy_query(model, device, pool_loader, query_size=query_size)
        
    elif query_strategy == 'bald':
        sample_idx = BALD_query(model, device, pool_loader, query_size = query_size, T = T, batch_size = batch_size)
        
    return sample_idx
    

def plot_decision_bound(model, moons_data):
    X, y = make_moons(n_samples = 1000, noise = 0.3, random_state = 9)
    x1 = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 500)
    x2 = np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 500)
    
    xx, yy = np.meshgrid(x1, x2)
    x_grid = np.column_stack((xx.ravel(), yy.ravel()))
    
    model.eval()

    softmax_out = F.softmax(model(torch.tensor(x_grid).float()), dim = 1)
    softmax_out = softmax_out.detach().numpy()[:,1].reshape(xx.shape)
    
    plt.figure(figsize = (8,4))
    plt.pcolormesh(xx, yy, softmax_out, cmap=plt.cm.RdBu_r, shading = 'auto')
    plt.scatter(X[:,0], X[:,1], c = y, cmap =plt.cm.RdBu_r, s = 8)
    plt.xlim(X[:,0].min()-0.5, X[:,0].max()+0.5)
    plt.ylim(X[:,1].min()-0.5, X[:,1].max()+0.5)
    plt.colorbar()
    plt.show()
    return

def get_softmax_grid(model, moons_data):
    X, y = moons_data
    x1 = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 500)
    x2 = np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 500)
    
    
    xx, yy = np.meshgrid(x1, x2)
    x_grid = np.column_stack((xx.ravel(), yy.ravel()))
    
    model.eval()

    softmax_out = F.softmax(model(torch.tensor(x_grid).float()), dim = 1)
    softmax_out = softmax_out.detach().numpy()[:,1].reshape(xx.shape)

    return X, y, xx, yy, softmax_out


def get_entropy_grid(model, moons_data):
    X, y = moons_data
    x1 = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 500)
    x2 = np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 500)
    
    xx, yy = np.meshgrid(x1, x2)
    x_grid = np.column_stack((xx.ravel(), yy.ravel()))
    
    model.eval()
    
    logits = model(torch.tensor(x_grid).float())
    entropy_out = e = -1.0 * torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), dim=1)
    entropy_out = entropy_out.detach().numpy().reshape(xx.shape)
    
    return X, y, xx, yy, entropy_out