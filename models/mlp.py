import torch
from torch import nn
import torch.nn.functional as F
class MLP (torch.nn.Module):

    def __init__(self, in_channels, hidden_channels=32, out_channels=2, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.in_channels = in_channels

        self.hidden1 = nn.Linear(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.hidden2 = nn.Linear(hidden_channels, hidden_channels//2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels//2)

        self.hidden3 = nn.Linear(hidden_channels//2, hidden_channels//4)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels//4)

        self.classifier = nn.Linear(hidden_channels//4, out_channels)

    def forward(self, x):
        x = F.relu(self.hidden1(x))  # activation function for hidden layer
        x = self.dropout(self.bn1(x))
        x = F.relu(self.hidden2(x))
        x = self.dropout(self.bn2(x))
        x = F.relu(self.hidden3(x))
        x = self.dropout(self.bn3(x))
        x = self.classifier(x)
        return x