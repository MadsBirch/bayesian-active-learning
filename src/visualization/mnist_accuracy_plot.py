import time 
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import os
os.chdir('/Users/madsbirch/Documents/bal')
print("Current working directory: {0}".format(os.getcwd()))

import pickle
import torch

from src.models.model import AL_Model, TwoMoons_Model, PaperCNN
from src.features.sampling import Sampling

from src.data.data import ActiveLearningDataset, get_balanced_sample_indices, get_targets
from src.data.data import TwoMoons, MNIST_CUSTOM


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.benchmark = True

# Set device
device = "mps" if torch.backends.mps.is_available() else "cpu"
#device = 'cpu'
print(f"Using device: {device}")

FIGURE_PATH = '/Users/madsbirch/Documents/bal/reports/figures/'

## mnist
mnist_train_data = MNIST_CUSTOM(root='', train=True)
mnist_test_data = MNIST_CUSTOM(root='', train=False)

# Generates the initial balanced pool.
# define how many classes there is in the data and how many samples per class you want.
n_classes = 10
n_init_samples = 20


initial_samples = get_balanced_sample_indices(
    get_targets(mnist_train_data),
    num_classes=n_classes,
    n_per_digit=n_init_samples / n_classes,
)

init_samples = {'idxs': initial_samples}
with open("mnist_init_samples", "wb") as fp:
    pickle.dump(init_samples, fp)
    
    
with open("/Users/madsbirch/Documents/bal/mnist_init_samples", 'rb') as f:
    init_samples_dict = pickle.load(f)

init_sample_idxs = init_samples_dict['idxs']

# parameters
lr = 1e-3
batch_size = 500
n_epochs = 150
query_size = 10
n_iter = 3
n_queries = 100
n_drop = 10
n_ens = 5

max_training_samples = 100

strat_list = ['random', 'entropy', 'margin', 'bald_mc']


# lists and dictionaries for storage of performance metrics
TEST_ACC = np.zeros((n_iter, n_queries+1))

test_dict = {
    'random': {'acc_mean': [], 'acc_se': [], 'bce_mean': [], 'bce_se': [],'dataset_len': []},
    'margin': {'acc_mean': [], 'acc_se': [], 'bce_mean': [], 'bce_se': [], 'dataset_len': []},
    'entropy': {'acc_mean': [], 'acc_se': [], 'bce_mean': [], 'bce_se': [], 'dataset_len': []},
    'bald_mc': {'acc_mean': [], 'acc_se': [], 'bce_mean': [], 'bce_se': [], 'dataset_len': []},
    'bald_ens': {'acc_mean': [], 'acc_se': [], 'bce_mean': [], 'bce_se': [], 'dataset_len': []},
}


for s in strat_list:
    print(s)
    #print(f'STRATEGY: {s}')
    for i in range(n_iter):
        #print(f'ITER: {i+1:2d}')
        
        torch.manual_seed(i)
        np.random.seed(i)
        random.seed(i)

        # define model and optimizer
        model = PaperCNN()
        al_model = AL_Model(model, device)

        # Initalize the Active Learning Dataset s.t. the training set is empty and the unlabaled pool is full.
        # Acquire the initial pool to the traindata.
        al_traindata = ActiveLearningDataset(mnist_train_data)
        al_traindata.acquire_samples(init_sample_idxs)

        # n_queries + 1 because of initial pool
        for q in tqdm(range(n_queries+1)):

            ## train modelx
            al_model.train(al_traindata.labeled_dataset, n_epochs=n_epochs, lr=lr)    
            
            ## test model
            acc = al_model.test(mnist_test_data)
            TEST_ACC[i,q] = acc
            
            # random way to 
            if i == 0:
                test_dict[s]["dataset_len"].append(len(al_traindata.labeled_dataset))
            
            if s == "random":
                batch = Sampling(al_traindata.unlabeled_dataset, al_model, query_size, device).random()
            
            if s == "margin":
                batch = Sampling(al_traindata.unlabeled_dataset, al_model, query_size, device).margin()

            if s == "entropy":
                batch = Sampling(al_traindata.unlabeled_dataset, al_model, query_size, device).entropy()
            
            if s == "bald_mc":
                batch = Sampling(al_traindata.unlabeled_dataset, al_model, query_size, device).bald_mc(n_drop=n_drop)
            
            if s == "bald_ens":
                batch = Sampling(al_traindata.unlabeled_dataset, al_model, query_size, device).bald_ensemble(traindata = al_traindata.labeled_dataset, n_ens = n_ens, n_epochs = n_epochs, lr = lr)
                
            al_traindata.acquire_samples(batch.indices)
                    
    test_dict[s]['acc_se'] = TEST_ACC.std(0)/np.sqrt(n_iter)
    test_dict[s]['acc_mean']  = TEST_ACC.mean(0)
 
with open(f'mnist_dict', "wb") as fp:
    pickle.dump(test_dict, fp)
    
    
with sns.axes_style("whitegrid"):
    fig, ax1 = plt.subplots(figsize=(10,8))  # Adjusted to create a single subplot

    x = np.linspace(start=n_init_samples,
                    stop=n_init_samples + n_queries * query_size, 
                    num=n_queries + 1, 
                    dtype=int)
    
    clrs = sns.color_palette("husl", len(strat_list))
    for i, s in enumerate(strat_list):
        acc_mean = test_dict[s]['acc_mean']
        acc_std = test_dict[s]['acc_se']
        ax1.plot(x, acc_mean, label=s, c=clrs[i])
        ax1.fill_between(x, acc_mean + acc_std, acc_mean - acc_std, alpha=0.2, facecolor=clrs[i])
        ax1.legend()
        ax1.set_title('Classification Accuracy')
        ax1.set_xlabel('Training set size')
        ax1.set_ylabel('Test Accuracy (%)')
        # ax1.tick_params('x', labelrotation=x_tick_rot)  # Uncomment if rotation is needed

    fig.savefig('reports/figures/MNIST_accuracy_curve.png')  # Save the figure
    fig.tight_layout()