from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.batchbald_redux import (
    active_learning,
    batchbald,
    consistent_mc_dropout,
    joint_entropy,
)

# MLP for the TwoMoons dataset
class MLP(nn.Module):
  '''
    Multilayer Perceptron for classification.
  '''
  def __init__(self, dropout = float):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(2, 50),
      nn.ReLU(),
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