import random
import numpy as np
import matplotlib.pyplot as plt

import pickle

import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler

from src.models.train_model import train, test
from src.features.oldacquistion_functions import random_query, query_the_oracle

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

FIGURE_PATH = '/Users/madsbirch/Documents/4_semester/BAL/bayesian-active-learning/reports/figures/'

def plot_acquisition_output(model,
                            traindata,
                            X_train,
                            y_train, 
                            grid_function, 
                            init_pool_idx, 
                            device,
                            strategy = str,
                            num_epochs = 200,
                            batch_size = 256,  
                            num_queries = 6, 
                            query_size = 3, 
                            lr = 1e-4,
                            train_T = 100,
                            plot_T = 30,
                            save_name = str):    
    # figure init
    fig, axs = plt.subplots(nrows=1, ncols=num_queries+1, figsize=(12, 2),sharex=True, sharey=True)
    mesh_alpha = 0.7

    j = 0

    # reset dataset, model and optimizer
    traindata.reset_mask()
    optimizer = optim.Adam(model.parameters(), lr = lr)

    # train on initial pool
    traindata.update_mask(init_pool_idx)
    labeled_loader = DataLoader(traindata, batch_size=batch_size, num_workers=0,
                                        sampler=SubsetRandomSampler(init_pool_idx), shuffle = False)

    model, _ = train(model, labeled_loader, optimizer, device, num_epochs=num_epochs, plot = False, printout = False)

    # plot initial 10 data points, first col in plot
    X, y, xx, yy, grid_out = grid_function(model, X_train, y_train, T = plot_T)

    mesh = axs[0].pcolormesh(xx, yy, grid_out, cmap=plt.cm.RdBu_r, alpha = mesh_alpha)
    axs[0].scatter(X[init_pool_idx,0], X[init_pool_idx,1], c = y[init_pool_idx], s=20)
    axs[0].scatter(X[:,0], X[:,1], c=y, s = 10, alpha = 0.15, marker = '.')
    axs[0].set_xlim(X[:,0].min()-0.5, X[:,0].max()+0.5)
    axs[0].set_ylim(X[:,1].min()-0.5, X[:,1].max()+0.5)

    labeled_idx_list = []

    for i, query in enumerate(range(num_queries)):
        # quering data points
        
        sample_idx = query_the_oracle(model, traindata, device, query_strategy=strategy, T=train_T, query_size=query_size)
        traindata.update_mask(sample_idx)
        labeled_idx = np.where(traindata.unlabeled_mask == 0)[0]
        labeled_idx_list.extend(sample_idx)
        
        # define a list for plotting the initial 10 points + the queried points
        plot_idx = init_pool_idx + labeled_idx_list      
                
        labeled_loader = DataLoader(traindata, batch_size=batch_size, num_workers=0,
                                        sampler=SubsetRandomSampler(labeled_idx), shuffle = False)

        # train model
        model = train(model, labeled_loader, optimizer, device, num_epochs=num_epochs, plot = False, printout = False)
        
        # plot grid and scatter for the rest of the cols.
        X, y, xx, yy, grid_out = grid_function(model, X_train, y_train, T = plot_T)

        mesh = axs[i+1].pcolormesh(xx, yy, grid_out, cmap=plt.cm.RdBu_r, alpha = mesh_alpha)
        axs[i+1].scatter(X[plot_idx,0], X[plot_idx,1], c = y[plot_idx], s = 20)
        axs[i+1].scatter(X[:,0], X[:,1], c=y, s = 10, alpha = 0.15, marker = '.')
        axs[i+1].set_xlim(X[:,0].min()-0.5, X[:,0].max()+0.5)
        axs[i+1].set_ylim(X[:,1].min()-0.5, X[:,1].max()+0.5)


    fig.tight_layout()
    fig.colorbar(mesh, ax=axs.ravel().tolist(), fraction=0.01, pad=0.01)

    # set ylabels to strategy
    axs[0].set_ylabel(strategy, fontsize=12)

    # set xlabels to number of sampled data points.
    ns = np.linspace(0, query_size*num_queries, num_queries+1, dtype = int)

    for i, n in enumerate(ns):
        axs[i].set_xlabel(f'n={int(len(init_pool_idx)+n)}', fontsize=12)

    plt.savefig(FIGURE_PATH+save_name+'.png')
    plt.show()