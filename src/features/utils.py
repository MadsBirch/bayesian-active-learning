from random import shuffle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.datasets import make_moons

import random

from src.models.train_model import train

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

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
            X, y, idx = batch
            logits = model(X.to(device))
            
            p_hat = F.softmax(logits, dim=1)
            e = (-1.0*p_hat * torch.log(p_hat)).sum(1)
            e_scores.extend(e.cpu().tolist())
            indices.extend(idx.tolist())

    conf = np.asarray(e_scores)
    ind = np.asarray(indices)
    sorted_pool = np.argsort(-1*conf)
    
    return ind[sorted_pool][:query_size] 
    #return indices, e_scores


def BALD_query(model, device, data_loader, data_len, X, y, batch_size, query_size = 10, T = 30, method = 'MC_drop'):

    x1 = np.linspace(X[:,0].min(), X[:,0].max(), 200)
    x2 = np.linspace(X[:,1].min(), X[:,1].max(), 200)
    
    xx, yy = np.meshgrid(x1, x2)
    x_grid = np.column_stack((xx.ravel(), yy.ravel()))

    logits_list = []
    indices = []
    Xs = []
    
    # logits is of dim n*c*T
    logits = torch.zeros((data_len,2,T))
    logits_grid = torch.zeros((len(x_grid),2,T))
    
    if method == 'MC_drop':
        model.train()
        for t in range(T):
            for i, batch in enumerate(data_loader):
                X, y, idx = batch
                logit = model(X.to(device))
                logits[i*len(y):i*len(y)+len(y),:,t] = logit    
                logits_grid[:,:,t] = model(torch.tensor(x_grid).float())
                
                if t == 0:
                    indices.extend(idx.tolist())
                    Xs.extend(X.tolist())
                                  
    
    if method == 'ensemble':
        
        lr = 6e-4
        for t in range(T):
            
            optimizer = optim.Adam(model.parameters(), lr = lr)
            model = train(model, data_loader, optimizer, device, num_epochs = 1000, plot = False, printout = False)
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    X, y, idx = batch
                    logit = model(X.to(device))
                    logits[i*len(y):i*len(y)+len(y),:,t] = logit
                    logits_grid[:,:,t] = model(torch.tensor(x_grid).float())
                    
                    if t == 0:
                        indices.extend(idx.tolist())
                        Xs.extend(X.tolist())
    
    # grid calc        
    var_grid = logits_grid.var(2).sum(1)
    var_grid = var_grid.detach().numpy().reshape(xx.shape)

    p_hat_grid = F.softmax(logits_grid, dim = 1)
    p_hat_mean_T_grid = p_hat_grid.mean(2)
    
    first_term_grid = (-p_hat_mean_T_grid*torch.log(p_hat_mean_T_grid)).sum(1)
    second_term_grid = (-p_hat_grid*torch.log(p_hat_grid)).sum(1).mean(1)
    
    BALD_scores_grid = first_term_grid-second_term_grid
    BALD_out_grid = BALD_scores_grid.detach().numpy().reshape(xx.shape)
    
    first_term_grid = first_term_grid.detach().numpy().reshape(xx.shape)
    second_term_grid = second_term_grid.detach().numpy().reshape(xx.shape)
    
    grids_list = [BALD_out_grid, var_grid, first_term_grid, second_term_grid]

    
    # data
    p_hat = F.softmax(logits, dim = 1)
    p_hat_mean_T= p_hat.mean(2)
    
    first_term = (-p_hat_mean_T*torch.log(p_hat_mean_T)).sum(1)
    second_term = (-p_hat*torch.log(p_hat)).sum(1).mean(1)
    BALD_scores = first_term - second_term
    
    return BALD_scores.sort()[1][-query_size:], xx, yy, grids_list

def query_the_oracle(model, dataset, device, 
                     T = 30,
                     query_size = 10, 
                     query_strategy = 'random', 
                     bald_method = 'MC_drop', 
                     batch_size = 100):
    
    unlabeled_idx = np.nonzero(dataset.unlabeled_mask)[0]
    data_len = len(unlabeled_idx)
    
    pool_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, 
                                sampler=SubsetRandomSampler(unlabeled_idx), shuffle=False)
    
    if query_strategy == 'random':
        sample_idx = random_query(pool_loader, query_size=query_size)
        return sample_idx
        
    elif query_strategy == 'margin':
        sample_idx = margin_query(model, device, pool_loader, query_size=query_size)
        return sample_idx
        
    elif query_strategy == 'least_conf':
        sample_idx = least_confidence_query(model, device, pool_loader, query_size=query_size)
        return sample_idx
        
    elif query_strategy == 'entropy':
        sample_idx = entropy_query(model, device, pool_loader, query_size=query_size)
        return sample_idx
        
    elif query_strategy == 'bald':
        sample_idx, scores, sorted_pool, Xs = BALD_query(model, device, pool_loader, data_len, query_size = query_size, T = T, batch_size = batch_size, method = bald_method)
        return sample_idx, scores, sorted_pool, Xs
    

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

def softmax_grid(model, X, y):
    x1 = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 500)
    x2 = np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 500)
    
    
    xx, yy = np.meshgrid(x1, x2)
    x_grid = np.column_stack((xx.ravel(), yy.ravel()))
    
    model.eval()

    softmax_out = F.softmax(model(torch.tensor(x_grid).float()), dim = 1)
    softmax_out = softmax_out.detach().numpy()[:,1].reshape(xx.shape)

    return xx, yy, softmax_out


def entropy_grid(model, X, y, T = int):
    x1 = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 500)
    x2 = np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 500)
    
    xx, yy = np.meshgrid(x1, x2)
    x_grid = np.column_stack((xx.ravel(), yy.ravel()))
    
    model.eval()
    
    logits = model(torch.tensor(x_grid).float())
    entropy_out = -1.0 * torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), dim=1)
    entropy_out = entropy_out.detach().numpy().reshape(xx.shape)
    
    return xx, yy, entropy_out


def BALD_grid(model, X, y, T = 20):

    x1 = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 500)
    x2 = np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 500)
    
    xx, yy = np.meshgrid(x1, x2)
    x_grid = np.column_stack((xx.ravel(), yy.ravel()))
    
    model.train()
    
    # BALD
    logits = torch.zeros((len(x_grid),2,T))
    for t in range(T):
        logits[:,:,t] = model(torch.tensor(x_grid).float())

    p_hat = F.softmax(logits, dim = 1)
    p_hat_mean_T = p_hat.mean(2)
    
    first_term = (-p_hat_mean_T*torch.log(p_hat_mean_T)).sum(1)
    second_term = (-p_hat*torch.log(p_hat)).sum(1).mean(1)
    
    BALD_scores = first_term-second_term    
    BALD_out = BALD_scores.detach().numpy().reshape(xx.shape)
    
    return xx, yy, BALD_out



## implement method
def BALD_grid_viz(model, X, y, T = 20, method = 'MC_drop'):

    x1 = np.linspace(X[:,0].min(), X[:,0].max(), 200)
    x2 = np.linspace(X[:,1].min(), X[:,1].max(), 200)
    
    xx, yy = np.meshgrid(x1, x2)
    x_grid = np.column_stack((xx.ravel(), yy.ravel()))
    
    if method == 'MC_drop':
        model.train()
        
        # BALD
        logits = torch.zeros((len(x_grid),2,T))
        for t in range(T):
            logits[:,:,t] = model(torch.tensor(x_grid).float())
        
        
    var = logits.var(2).sum(1)
    var = var.detach().numpy().reshape(xx.shape)

    p_hat = F.softmax(logits, dim = 1)
    p_hat_mean_T = p_hat.mean(2)
    
    first_term = (-p_hat_mean_T*torch.log(p_hat_mean_T)).sum(1)
    second_term = (-p_hat*torch.log(p_hat)).sum(1).mean(1)
    
    BALD_scores = first_term-second_term
    BALD_out = BALD_scores.detach().numpy().reshape(xx.shape)
    
    first_term = first_term.detach().numpy().reshape(xx.shape)
    second_term = second_term.detach().numpy().reshape(xx.shape)
    
    grids_list = [BALD_out, var, first_term, second_term]
    
    return xx, yy, grids_list