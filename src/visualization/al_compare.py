import argparse
from multiprocessing import pool
import sys
from tqdm import tqdm

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
import torchvision.transforms as transforms

from src.models.model import MLP, PaperCNN
from src.data.data import TwoMoons, MNIST_CUSTOM
from src.models.train_model import train, test
from src.features.acquistion_funcs import query_the_oracle
from torchvision import datasets


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

class CompareAcquisitionFunctions(object):
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

    def plot_curve(self):
        print(f'Plotting performance on test set over increasing number of training samples...')
        parser = argparse.ArgumentParser(description='BALD arguments')
        parser.add_argument('--strat_list', nargs='+', default=['bald', 'entropy', 'random'])
        parser.add_argument('--dataset', default= 'MNIST', type = str)
        parser.add_argument('--n_iter', default=3, type=int)
        parser.add_argument('--num_queries', default=10, type=int)
        parser.add_argument('--query_size', default=10, type=int)
        parser.add_argument('--bald_method', default='MC_drop', type=str)
        parser.add_argument('--T', default=10, type=int)
        parser.add_argument('--dropout', default=0.3, type=float)
        parser.add_argument('--init_pool_size', default=20, type=int)
        parser.add_argument('--save_name', default='al_compare', type=str)
        parser.add_argument('--device', default='cpu', type=str)
        parser.add_argument('--subset', default=True, type=bool)
        
        args = parser.parse_args(sys.argv[2:])
        
        print(f"Using device: {args.device}")
        
        FIGURE_PATH = '/Users/madsbirch/Documents/4_semester/BAL/bayesian-active-learning/reports/figures/al_compare/'
        MODEL_PATH = '/Users/madsbirch/Documents/4_semester/BAL/bayesian-active-learning/models/'
        
        TEST_ACC = np.zeros((args.n_iter, args.num_queries+1))
        query_dict = {
            'bald': {'acc_mean': [], 'acc_se': []},
            'random': {'acc_mean': [], 'acc_se': []},
            'entropy': {'acc_mean': [], 'acc_se': []},
            'margin': {'acc_mean': [], 'acc_se': []}
        }

        # define model
        if args.dataset == 'TwoMoons':
            
            num_epochs = 1000
            batch_size = 256
            lr = 6e-4
        
            model = MLP(dropout=args.dropout)
            optimizer = optim.Adam(model.parameters(), lr = lr)

            
            # generate data
            X, y = make_moons(n_samples = 1000, noise = 0.2, random_state=9)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

            traindata = TwoMoons(X_train, y_train, return_idx = True)
            testdata = TwoMoons(X_test, y_test, return_idx = True)
            testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=0)
        
            num_samples = 100
            random_indices = torch.randperm(num_samples)
            valdata = Subset(testdata, random_indices)
            
            valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, num_workers=0)
            
            # generate a balanced inital pool
            initial_idx = np.array([],dtype=int)
            for i in range(2):
                idx = np.random.choice(np.where(traindata.y ==i)[0], size=5, replace=False)
                initial_idx = np.concatenate((initial_idx, idx))
                
        if args.dataset == 'MNIST':            
            num_epochs = 50
            batch_size = 256
            lr = 1e-3
            
            model = PaperCNN()
            optimizer = optim.Adam(model.parameters(), lr = lr)
            
            # train and test data
            traindata = MNIST_CUSTOM(root='data/raw', train = True, transform = transforms.ToTensor())
            testdata = MNIST_CUSTOM(root='data/raw', train = False, transform = transforms.ToTensor())
            
            # generate a balanced inital pool
            initial_idx = []
            for i in range(10):
                initial_idx.extend(np.random.choice(np.where(traindata.targets==i)[0], size=2, replace=False))
                
            # use subset if specified and make sure the initial pool is included
            if args.subset:
                num_samples = 1500
                train_idxs = np.arange(0,1500, 1) #np.random.randint(0, len(traindata), size = num_samples).tolist()

            # generate validation set of 100 samples
            num_samples = 100
            random_indices = torch.randperm(num_samples)
            valdata = Subset(testdata, random_indices)
            
            # dataloaders
            trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=False, num_workers=0)
            valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, num_workers=0)
            testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=0)
                      
        # train model on initial pool and save to disc (load in later)
        print(f'Training common model on initial pool...')
        
        # train on initial labeled pool and save model
        labeled_subset = Subset(traindata, initial_idx)
        labeled_loader = DataLoader(labeled_subset, batch_size=batch_size, num_workers=0, shuffle = False)
        model, optimizer = train(model, labeled_loader, optimizer, args.device, num_epochs=num_epochs, val = False, plot = False, printout = False)

        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, MODEL_PATH+args.dataset+'model.pth')

        for s in args.strat_list:
            print(f'STRATEGY: {s}')
            for i in range(args.n_iter):
                print(f'ITER: {i+1:2d}')
                torch.manual_seed(i)
                
                unlab_idxs = train_idxs
                # load model trained on initial pool
                if args.dataset == 'MNIST':
                    model = PaperCNN()
                    
                if args.dataset == 'TwoMoons':
                    model = MLP(dropout=args.dropout)
                    
                optimizer = optim.Adam(model.parameters(), lr = lr)

                state = torch.load(MODEL_PATH+args.dataset+'model.pth')
                model.load_state_dict(state['state_dict'])
                optimizer.load_state_dict(state['optimizer'])
                
                TEST_ACC[i,0] = test(model, testloader, args.device, display = False)
                
                lab_pool_idxs = initial_idx
                
                for query in tqdm(range(args.num_queries)):
                    

                    # quering data points            
                    unlab_pool_subset = Subset(traindata, unlab_idxs)
                    unlab_pool_loader = DataLoader(unlab_pool_subset, batch_size=batch_size, num_workers=0, shuffle = False)
                    sample_idx = query_the_oracle(model, unlab_pool_loader, 
                                                            args.device,
                                                            T = args.T,
                                                            query_size=args.query_size,
                                                            query_strategy= s,
                                                            bald_method=args.bald_method
                                                            )
                    
                    unlab_idxs = [x for x in unlab_idxs if (x != sample_idx).all()]
                    lab_pool_idxs.extend(sample_idx)
                    
                    # train on labeled pool subset
                    labeled_subset = Subset(traindata, lab_pool_idxs)
                    labeled_loader = DataLoader(labeled_subset, batch_size=batch_size, num_workers=0,shuffle = False)
                    model, optimizer = train(model, labeled_loader, optimizer, args.device, valloader, num_epochs=num_epochs, val = False, plot = False, printout = False)

                    # test model
                    TEST_ACC[i,query+1] = test(model, testloader, args.device, display = False)

            query_dict[s]['acc_se'] = TEST_ACC.std(0)/np.sqrt(args.n_iter)
            query_dict[s]['acc_mean']  = TEST_ACC.mean(0)

        x = np.linspace(start = args.init_pool_size,
                    stop = args.num_queries*args.query_size, 
                    num = args.num_queries+1, 
                    dtype=int)
        
        plt.figure(figsize=(10,6))
        for s in args.strat_list:
            mean = query_dict[s]['acc_mean']
            std = query_dict[s]['acc_se']
            plt.plot(x, mean, label = s)
            plt.legend()
            plt.title('Performance of Active Learning w. Different Acquisition Strategies')
            plt.xlabel('Number of Traning Points')
            plt.ylabel('Test Accuracy (%)')
            plt.xticks(x, x)
            plt.fill_between(x, mean+std, mean-std, alpha = 0.4)
        plt.savefig(FIGURE_PATH+args.dataset+args.save_name+'.png')
        plt.show()
        
        
    def plot_grid(self):
        print(f'Plotting decision boundary...')
        parser = argparse.ArgumentParser(description='BALD arguments')
        parser.add_argument('--strat_list', default= ['random', 'margin', 'entropy', 'bald'], type =list)
        parser.add_argument('--n_iter', default=5, type=int)
        parser.add_argument('--num_queries', default=20, type=int)
        parser.add_argument('--query_size', default=5, type=int)
        parser.add_argument('--bald_method', default='MC_drop', type=str)
        parser.add_argument('--T', default=100, type=int)
        parser.add_argument('--dropout', default=0.1, type=float)
        parser.add_argument('--init_pool_size', default=10, type=int)
        parser.add_argument('--save_name', default='al_compare', type=str)
        parser.add_argument('--device', default='cpu', type=str)
        
        args = parser.parse_args(sys.argv[1:])
        
        print(f"Using device: {args.device}")
        
        FIGURE_PATH = '/Users/madsbirch/Documents/4_semester/BAL/bayesian-active-learning/reports/figures/al_compare/'
        MODEL_PATH = '/Users/madsbirch/Documents/4_semester/BAL/bayesian-active-learning/models/'
            

if __name__ == '__main__':
    CompareAcquisitionFunctions()
