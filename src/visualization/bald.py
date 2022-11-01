import argparse
from multiprocessing import pool
import sys

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.models.model import MLP
from src.data.data import get_dataloaders, TwoMoons
from src.models.train_model import train, test
from src.features.utils import softmax_grid, BALD_query, query_the_oracle

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

class PlotBALD(object):
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for plotting the BALD acquisition function",
            usage="python bald.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def plot(self):
        print("Plotting BALD...")
        parser = argparse.ArgumentParser(description='BALD arguments')
        parser.add_argument('--method', default='MC_drop', type=str)
        parser.add_argument('--T', default=10, type=int)
        parser.add_argument('--dropout', default=0.1, type=float)
        parser.add_argument('--num_queries', default=4, type=int)
        parser.add_argument('--query_size', default=5, type=int)
        parser.add_argument('--init_pool_size', default=10, type=int)
        parser.add_argument('--save_name', default='bald', type=str)
        parser.add_argument('--device', default='cpu', type=str)
        
        args = parser.parse_args(sys.argv[2:])
        
        print(f"Using device: {args.device}")
        
        FIGURE_PATH = '/Users/madsbirch/Documents/4_semester/BAL/bayesian-active-learning/reports/figures/bald/'

        # generate data
        X, y = make_moons(n_samples = 1000, noise = 0.2, random_state=9)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

        traindata = TwoMoons(X_train, y_train, return_idx = True)
        testdata = TwoMoons(X_test, y_test, return_idx = True)
                
        # define model
        num_epochs = 1000
        batch_size = 256
        lr = 6e-4
        
        init_pool_idx = np.random.randint(0,500, size = args.init_pool_size).tolist()
        print(f'Initial pool size {len(init_pool_idx)}')
        
        # figure init
        label_list = ['BALD', 'var', 'first_term', 'second_term', 'softmax']
        fig, axs = plt.subplots(nrows=len(label_list), ncols=args.num_queries+1, figsize=(10, 10),sharex=True, sharey=True)
        mesh_alpha = 0.8
        
        # reset dataset
        traindata.reset_mask()
        traindata.update_mask(init_pool_idx)
        
        #  model and optimizer
        model = MLP(dropout=args.dropout)
        optimizer = optim.Adam(model.parameters(), lr = lr)
        
        # evaluate BALD on unlabeled pool of data
        sample_idx, xx, yy, grids_list = query_the_oracle(model, traindata, args.device, X_train, y_train,
                                                          T = args.T,
                                                          query_size=args.query_size,
                                                          query_strategy= 'bald',
                                                          bald_method=args.method,
                                                          batch_size=batch_size)

        xx_soft, yy_soft, softmax_out = softmax_grid(model, X_train, y_train)

        for i, g in enumerate(grids_list):
            mesh = axs[i,0].pcolormesh(xx, yy, g, cmap=plt.cm.RdBu_r, alpha = mesh_alpha)
            axs[i,0].scatter(X_train[init_pool_idx,0], X_train[init_pool_idx,1], c= y_train[init_pool_idx], marker = 'X')
            axs[i,0].scatter(X_train[:,0], X_train[:,1], c = y_train, alpha = 0.1, marker = '.')
            axs[i,0].set_xlim(X_train[:,0].min(), X_train[:,0].max())
            axs[i,0].set_ylim(X_train[:,1].min(), X_train[:,1].max())
            
        mesh = axs[4,0].pcolormesh(xx_soft, yy_soft, softmax_out, cmap=plt.cm.RdBu_r, alpha = mesh_alpha)
        axs[4,0].scatter(X_train[init_pool_idx,0], X_train[init_pool_idx,1], c = y_train[init_pool_idx], marker = 'X')
        axs[4,0].scatter(X_train[:,0], X_train[:,1], c = y_train, alpha = 0.1,marker = '.')
        axs[4,0].set_xlim(X_train[:,0].min(), X_train[:,0].max())
        axs[4,0].set_ylim(X_train[:,1].min(), X_train[:,1].max())

        # train on initial labeled pool
        labeled_subset = Subset(traindata, init_pool_idx)
        labeled_loader = DataLoader(labeled_subset, batch_size=batch_size, num_workers=0, shuffle = False)
        model, optimizer = train(model, labeled_loader, optimizer, args.device, valloader = None, num_epochs=num_epochs, val = False, plot = False, printout = False)
        
        for j, query in enumerate(range(args.num_queries)):
            print(f'// Query {j+1:2d} of {args.num_queries}')

            # evaluate BALD on unlabeled pool of data
            sample_idx, xx, yy, grids_list = query_the_oracle(model, traindata, args.device, X_train, y_train,
                                                                T = args.T,
                                                                query_size=args.query_size,
                                                                query_strategy= 'bald',
                                                                bald_method=args.method,
                                                                batch_size=batch_size)

            xx_soft, yy_soft, softmax_out = softmax_grid(model, X_train, y_train)
            
            # update labeled pool of data
            traindata.update_mask(sample_idx)
            labeled_idx = np.where(traindata.unlabeled_mask == 0)[0]

            for k, g in enumerate(grids_list):
                mesh = axs[k,j+1].pcolormesh(xx, yy, g, cmap=plt.cm.RdBu_r, alpha = mesh_alpha)
                axs[k,j+1].scatter(X_train[labeled_idx,0], X_train[labeled_idx,1], c = y_train[labeled_idx], marker = 'X')
                axs[k,j+1].scatter(X_train[:,0], X_train[:,1], c = y_train, alpha = 0.1, marker = '.')
                axs[k,j+1].set_xlim(X_train[:,0].min(), X_train[:,0].max())
                axs[k,j+1].set_ylim(X_train[:,1].min(), X_train[:,1].max())
            
            mesh = axs[4,j+1].pcolormesh(xx_soft, yy_soft, softmax_out, cmap=plt.cm.RdBu_r, alpha = mesh_alpha)
            axs[4,j+1].scatter(X_train[labeled_idx,0], X_train[labeled_idx,1], c = y_train[labeled_idx], marker = 'X')
            axs[4,j+1].scatter(X_train[:,0], X_train[:,1], c = y_train, alpha = 0.1, marker = '.')
            axs[4,j+1].set_xlim(X_train[:,0].min(), X_train[:,0].max())
            axs[4,j+1].set_ylim(X_train[:,1].min(), X_train[:,1].max())
              
            # train on labeled pool subset
            labeled_subset = Subset(traindata, labeled_idx)
            labeled_loader = DataLoader(labeled_subset, batch_size=batch_size, num_workers=0,shuffle = False)
            model, optimizer = train(model, labeled_loader, optimizer, args.device, valloader = None, num_epochs=num_epochs, val = False, plot = False, printout = False)

        fig.colorbar(mesh, ax=axs.ravel().tolist(), fraction=0.01, pad=0.01)

        # set ylabels to strategy
        for i, label in enumerate(label_list):
            axs[i,0].set_ylabel(label, fontsize=12)

        # set xlabels to number of sampled data points.
        ns = np.linspace(0, args.query_size*args.num_queries, args.num_queries+1, dtype=int)

        for i, n in enumerate(ns):
            axs[len(label_list)-1,i].set_xlabel(f'n={int(args.init_pool_size+n)}', fontsize=12)

        plt.savefig(FIGURE_PATH+args.save_name+'_'+args.method+'_T'+str(args.T)+'_drop'+str(args.dropout)+'.png')
        plt.show()
        
if __name__ == '__main__':
    PlotBALD()
