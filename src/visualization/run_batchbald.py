import math
import random
import numpy as np
import os
from dataclasses import dataclass
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from tqdm import tqdm
import pickle

from src.models.model import BayesianCNN

import torch
from torch import nn as nn
from torch.nn import functional as F

from src.batchbald_redux import (
    active_learning,
    batchbald,
    consistent_mc_dropout,
    joint_entropy,
    repeated_mnist,
)

test_dict = {
    'acc': [],
    'loss': [],
    'dataset_len': []
}


use_cuda = torch.cuda.is_available()
print(f"use_cuda: {use_cuda}")
device = "cuda" if use_cuda else "cpu"
kwargs = {"num_workers": 0, "pin_memory": True} if use_cuda else {}


algo_list = ["batchbald"]
final_test_accs = []
final_indices = []

max_training_samples = 50  # Maximum limit of train samples needed
num_inference_samples = 10
num_test_inference_samples = 5
num_samples = 100000  # Total number of samples

test_batch_size = 512  # Test Loader Batch size
batch_size = 64  # Train loader Batch size
scoring_batch_size = 128  # Pool Loader Batch size
training_iterations = 4096 * 6

acquisition_batch_size = 4


seed_value = 0

torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
os.environ["PYTHONHASHSEED"] = str(seed_value)

num_initial_samples = 20  # Number of initial samples required
num_classes = 10  # Total classes in MNIST dataset

# get an active learning dataset
train_dataset = MNIST(root='./data/raw', train=True, download = True, transform=transforms.ToTensor())
test_dataset = MNIST(root='./data/raw', train=False, download = True, transform=transforms.ToTensor())

# Generates 20 samples (2 from each class) and returns their indices
initial_samples = active_learning.get_balanced_sample_indices(
    repeated_mnist.get_targets(train_dataset),
    num_classes=num_classes,
    n_per_digit=num_initial_samples / num_classes,
)

test_accs = []
test_loss = []
added_indices = []

active_learning_data = active_learning.ActiveLearningData(
    train_dataset
)  # Splits the dataset into training dataset and pool dataset

active_learning_data.acquire(
    initial_samples
)  # Seperates the initial indices from the pool and fixes it as initial train dataset

train_loader = torch.utils.data.DataLoader(
    active_learning_data.training_dataset,
    sampler=active_learning.RandomFixedLengthSampler(
        active_learning_data.training_dataset, training_iterations
    ),
    batch_size=batch_size,
    **kwargs,
)

pool_loader = torch.utils.data.DataLoader(
    active_learning_data.pool_dataset,
    batch_size=scoring_batch_size,
    shuffle=False,
    **kwargs,
)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)

pbar = tqdm(
    initial=len(active_learning_data.training_dataset),
    total=max_training_samples,
    desc="Training Set Size",
)

model = BayesianCNN(num_classes).to(device=device)  # initialise model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


while True:    
    test_dict["dataset_len"].append(len(active_learning_data.training_dataset))

    model.train()
    # Train
    for data, target in tqdm(train_loader, desc="Training", leave=False):
        data = data.to(device=device)
        #assert data.device == torch.device("cuda")
        target = target.to(device=device)

        optimizer.zero_grad()

        prediction = model(data, 1).squeeze(1)
        loss = F.nll_loss(prediction, target)

        loss.backward()
        optimizer.step()

    # Test
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing", leave=False):
            data = data.to(device=device)
            target = target.to(device=device)

            prediction = torch.logsumexp(model(data, num_test_inference_samples), dim=1) - math.log(
                num_test_inference_samples
            )
            loss += F.nll_loss(prediction, target, reduction="sum")

            prediction = prediction.max(1)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    loss /= len(test_loader.dataset)
    test_dict["loss"].append(loss)

    percentage_correct = 100.0 * correct / len(test_loader.dataset)
    test_dict["acc"].append(percentage_correct)

    print("Test set: Average loss: {:.4f}, Accuracy: ({:.2f}%)".format(loss, percentage_correct))

    if len(active_learning_data.training_dataset) >= max_training_samples:
        break

    # Acquire pool predictions
    N = len(active_learning_data.pool_dataset)
    logits_N_K_C = torch.empty(
        (N, num_inference_samples, num_classes),
        dtype=torch.double,
        pin_memory=use_cuda,
    )

    with torch.no_grad():
        model.eval()

        for i, (data, _) in enumerate(tqdm(pool_loader, desc="Evaluating Acquisition Set", leave=False)):
            data = data.to(device=device)

            lower = i * pool_loader.batch_size
            upper = min(lower + pool_loader.batch_size, N)
            logits_N_K_C[lower:upper].copy_(model(data, num_inference_samples), non_blocking=True)

    with torch.no_grad():
        candidate_batch = batchbald.get_batchbald_batch(
            logits_N_K_C,
            acquisition_batch_size,
            num_samples,
            dtype=torch.double,
            device=device,  # Returns the indices and scores(Mutual Information) for the batch selected by Batchbald/BALD Strategy.
        )

    targets = repeated_mnist.get_targets(active_learning_data.pool_dataset)  # Returns the target labels
    dataset_indices = active_learning_data.get_dataset_indices(
        candidate_batch.indices
    )  # Returns indices for candidate batch

    print("Dataset indices: ", dataset_indices)
    print("Scores: ", candidate_batch.scores)
    print("Labels: ", targets[candidate_batch.indices])

    active_learning_data.acquire(candidate_batch.indices)  # add the new indices to training dataset
    added_indices.append(dataset_indices)
    pbar.update(len(dataset_indices))

final_test_accs.append(test_accs)
final_indices.append(added_indices)

with open("test_dict", "wb") as fp:
    pickle.dump(test_dict, fp)
