from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchmetrics import CalibrationError

from tqdm import tqdm
import numpy as np

"""""
from src.batchbald_redux import (
    active_learning,
    batchbald,
    consistent_mc_dropout,
    joint_entropy,
)
"""""

class AL_Model():
  def __init__(self, model, device):
        self.model = model
        self.device = device
  
  def train(self, data, n_epochs = 50, lr = 1e-3):
    
    self.lr = lr
    self.n_epochs = n_epochs
    self.model.to(self.device)
    self.model.train()
    
    optimizer = optim.Adam(self.model.parameters(), lr = lr)
    loader = DataLoader(data, shuffle=False, batch_size=256)
    
    for epoch in range(n_epochs):
      for x, y, idxs in loader:
        x, y = x.to(self.device), y.to(self.device)
        optimizer.zero_grad()
        out = self.model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
  
  def predict(self, data):
    
    self.model.eval()
    preds = torch.zeros(len(data), dtype=data.targets.dtype)
    loader = DataLoader(data, shuffle=False, batch_size=256)
    with torch.no_grad():
        for x, y, idxs in loader:
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x)
            pred = out.max(1)[1]
            preds[idxs] = pred.cpu()
    return preds
  
    
  def predict_probs(self, data):
    
    self.model.eval()
    
    probs = []
    loader = DataLoader(data, shuffle=False, batch_size=500)
        
    with torch.no_grad():
      for x, y, idxs in loader:
        x, y = x.to(self.device), y.to(self.device)
        out = self.model(x)
        probs.append(F.softmax(out, dim=1).cpu())

    probs = torch.cat(probs)

    return probs
  
  
  def predict_probs_mc(self, data, n_drop=10):
    probs = []
    loader = DataLoader(data, shuffle=False, batch_size=500)

    self.model.train()  # Ensure dropout is enabled
    with torch.no_grad():
        for t in range(n_drop):
            outputs_inner = []
            for batch in loader:
                X, _,_ = batch  
                outputs_inner.append(F.softmax(self.model(X.to(self.device)), dim=1))
            probs.append(torch.cat(outputs_inner, dim=0))

    probs = torch.stack(probs, dim=-1)
    return probs

  def predict_probs_ensemble(self, data, traindata, n_ens=5, n_epochs=50, lr=1e-3):
    probs = []
    loader = DataLoader(data, shuffle=False, batch_size=500)
    
    for t in range(n_ens):
        outputs_inner = []
        torch.manual_seed(t) 
        self.train(traindata, n_epochs=n_epochs, lr=lr)  # Train each model
        self.model.eval()  
        
        with torch.no_grad():
            for batch in loader:
                X, _,_ = batch 
                outputs_inner.append(F.softmax(self.model(X.to(self.device)), dim=1))
        probs.append(torch.cat(outputs_inner, dim=0))
   
    probs = torch.stack(probs, dim=-1) 
    return probs
  
  def test(self, data, bce_task = 'binary'):
      
    ## accuracy
    preds = self.predict(data)
    acc = (preds == data.targets).sum().item()/len(data)*100
    
    ## BCE
    #ece_metric = CalibrationError(n_bins = 10, task = bce_task) # define metric
    
    ### get model predictions
    probs = self.predict_probs(data)
    probsmax, _ = torch.max(probs, dim = 1)
    
    #ece = ece_metric(probsmax, data.targets)
    
    return acc
    
    
  

# Model for the TwoMoons dataset
class TwoMoons_Model(nn.Module):
  '''
    Multilayer Perceptron for classification.
  '''
  def __init__(self, dropout = float):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(2, 50),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(50, 50),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(50, 2)
    )
  def forward(self, x):
    return self.layers(x)


# MNIST CNN implemented in the paper https://arxiv.org/abs/1703.02910
class PaperCNN(nn.Module):
    def __init__(self, in_channels = 1, n_classes=10) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        self.lin =  nn.Sequential(
            nn.Linear(11*11*32,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.lin(x)
        return x

"""""
class BayesianCNN(consistent_mc_dropout.BayesianModule):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv1_drop = consistent_mc_dropout.ConsistentMCDropout2d()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = consistent_mc_dropout.ConsistentMCDropout2d()
        self.fc1 = nn.Linear(1024, 128)
        self.fc1_drop = consistent_mc_dropout.ConsistentMCDropout()
        self.fc2 = nn.Linear(128, num_classes)

    def mc_forward_impl(self, input: torch.Tensor):
        input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
        input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))
        input = input.view(-1, 1024)
        input = F.relu(self.fc1_drop(self.fc1(input)))
        input = self.fc2(input)
        input = F.log_softmax(input, dim=1)

        return input
"""""