import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, D_in, D_out, H=25):
        super(DQN, self).__init__()
        self.H = H
        self.fc1 = nn.Linear(D_in, self.H)
        self.fc2 = nn.Linear(self.H,self.H)
        self.fc3 = nn.Linear(self.H, D_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim = -1)
        return x