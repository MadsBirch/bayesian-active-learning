from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
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

class CNN(nn.Module):
    def __init__(self, dropout = float):
        super().__init__()
        self.conv = nn.Sequential(
          nn.Conv2d(3, 6, 5),
          nn.ReLU(),
          nn.Dropout2d(dropout),
          nn.MaxPool2d(2, 2),
          nn.Conv2d(6, 16, 5),
          nn.ReLU(),
          nn.Dropout2d(dropout),
          )
        self.linear = nn.Sequential(
          nn.Linear(16 * 5 * 5, 120),
          nn.ReLU(),
          nn.Dropout(dropout),
          nn.Linear(120, 84),
          nn.ReLU(),
          nn.Dropout(dropout),
          nn.Linear(84, 10)  
        )
    def forward(self, x):
      x = self.conv(x)
      x = torch.flatten(x, 1) # flatten all dimensions except batch
      x = self.linear(x)
      return x
    
    
class Net(nn.Module):
    def __init__(self, dropout = float):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        self.drop = nn.Dropout(dropout)
        self.drop_2d = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.pool(self.drop_2d(F.relu(self.conv1(x))))
        x = self.pool(self.drop_2d(F.relu(self.conv2(x))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x