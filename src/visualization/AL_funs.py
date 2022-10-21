import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons, make_circles, make_classification
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split

import pickle

import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler


from src.models.model import MLP
from src.data.data import get_dataloaders, TwoMoons
from src.models.train_model import train, test
from src.features.utils import query_the_oracle, BALD_query

torch.manual_seed(0)
np.random.seed(1)
random.seed(0)

# Set device
device = "mps" if torch.backends.mps.is_available() else "cpu"
device = 'cpu'
print(f"Using device: {device}")

FIGURE_PATH = '/Users/madsbirch/Documents/4_semester/BAL/bayesian-active-learning/reports/figures/'


# generate data
X, y = make_moons(n_samples = 1000, noise = 0.2, random_state=9)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

traindata = TwoMoons(X_train, y_train, return_idx = True)
testdata = TwoMoons(X_test, y_test, return_idx = True)

# dataloader and dataset class
train_loader, test_loader = get_dataloaders(traindata, testdata, batch_size=256)

# model, optimizer nad hyper parameters'
drop_out = 0.1

model = MLP(drop_out=drop_out)
num_epochs = 1000
batch_size = 256

# acrtive learning params
num_queries = 20
query_size = 10
lr = 6e-4

# posterior approx params
T = 1000

# list of sampling strategies
strat_list = ['random', 'margin', 'entropy', 'bald']

init_pool_size = 10
init_pool_idx = np.random.randint(0,500, size = init_pool_size).tolist()


query_dict = {
    'bald': {'acc_mean': [], 'acc_se': []},
    'random': {'acc_mean': [], 'acc_se': []},
    'entropy': {'acc_mean': [], 'acc_se': []},
    'margin': {'acc_mean': [], 'acc_se': []}
}

## setup ##
save_dict = False
plot = True

n_iter = 5
TEST_ACC = np.zeros((n_iter, num_queries+1))

bald_method = 'ensemble'


# seed list
for s in strat_list:
    #torch.manual_seed(r)
    
    print(f'STRATEGY: {s}')
    for i in range(n_iter):
        print(f'ITER: {i+1:2d}')
        
        torch.manual_seed(i)
        # reset dataset, model and optimizer
        traindata.reset_mask()
        model = MLP(drop_out=drop_out)
        optimizer = optim.Adam(model.parameters(), lr = lr)
        
        # train on initial pool
        traindata.update_mask(init_pool_idx)
        labeled_loader = DataLoader(traindata, batch_size=batch_size, num_workers=0,
                                        sampler=SubsetRandomSampler(init_pool_idx), shuffle = False)
    
        model = train(model, labeled_loader, optimizer, device, num_epochs=num_epochs, plot = False, printout = False)
        TEST_ACC[i,0] = test(model, test_loader, device, display = False)
    
        for query in range(num_queries):
            # quering data points
            if s == 'bald':
                sample_idx, scores, all_scores, Xs = query_the_oracle(model, traindata, device, query_strategy=s, query_size=query_size, T = T, batch_size=batch_size, method = bald_method)
            else:
                sample_idx = query_the_oracle(model, traindata, device, query_strategy=s, query_size=query_size, T = T, batch_size=batch_size)
            
            traindata.update_mask(sample_idx)
            labeled_idx = np.where(traindata.unlabeled_mask == 0)[0]
            labeled_loader = DataLoader(traindata, batch_size=batch_size, num_workers=0,
                                        sampler=SubsetRandomSampler(labeled_idx), shuffle=False)

            # train model
            model = train(model,labeled_loader,optimizer, device, num_epochs=num_epochs, plot = False, printout = False)

            # test model
            test_acc = test(model, test_loader, device, display = False)
            TEST_ACC[i,query+1] = test_acc

            print(f'# Samples: {init_pool_size + query*query_size:3d} | Test accururacy : {test_acc:.2f}%')

    query_dict[s]['acc_se'] = TEST_ACC.std(0)/np.sqrt(n_iter)
    query_dict[s]['acc_mean']  = TEST_ACC.mean(0)
    
if save_dict:
    with open("QUERY_DICT_50", "wb") as fp:
        pickle.dump(query_dict, fp)

if plot:
    x = np.arange(init_pool_size,num_queries+init_pool_size+1,dtype=int)
    plt.figure(figsize=(10,6))

    for s in strat_list:
        mean = query_dict[s]['acc_mean']
        std = query_dict[s]['acc_se']
        plt.plot(x, mean, label = s)
        plt.legend()
        plt.title('Performance of Active Learning w. Different Acquisition Strategies')
        plt.xlabel('Number of Traning Points')
        plt.ylabel('Test Accuracy (%)')
        plt.xticks(x, x)
        plt.fill_between(x, mean+std, mean-std, alpha = 0.4)
    plt.savefig('reports/figures/TwoMoons_results_13.png')
    plt.show()