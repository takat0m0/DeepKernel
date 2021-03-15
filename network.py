# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3) # 28x28x32 -> 26x26x32
        self.conv2 = nn.Conv2d(32, 64, 3) # 26x26x64 -> 24x24x64 
        self.pool = nn.MaxPool2d(2, 2) # 24x24x64 -> 12x12x64
        self.fc1 = nn.Linear(12 * 12 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = self.pool(F.relu(self.conv2(h)))
        h = h.view(-1, 12 * 12 * 64)
        h = F.relu(self.fc1(h))
        logit = self.fc2(h)
        return logit        
