import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
  '''
    Multilayer Perceptron for classification.
  '''
  def __init__(self, drop_out = float):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(2, 50),
      nn.Dropout(drop_out),
      nn.ReLU(),
      nn.Linear(50, 50),
      nn.Dropout(drop_out),
      nn.ReLU(),
      nn.Linear(50, 2)
      #nn.Softmax(dim = 0)
    )

  def forward(self, x):
    return self.layers(x)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
