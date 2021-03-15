# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from sample_data import gt, train_data

class RegressionNet(nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__()
        self.fc1 = nn.Linear(1, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 100)                
        self.fc6 = nn.Linear(100, 1)        

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))        
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(h))
        h = self.fc6(h)
        return h

if __name__ == '__main__':
    model = RegressionNet()
    
    data_num = 30
    input_dim = 1
    
    gt_x, gt_y = gt()
    x, y = train_data(data_num, input_dim)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    print(x.shape, y.shape)

    torch_x = torch.tensor(x)
    torch_y = torch.tensor(y)

    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    mse_loss = nn.MSELoss()
    
    for i in range(10000):
        model.train()
        optimizer.zero_grad()
        output = model(torch_x)
        loss = mse_loss(output, torch_y)
        loss.backward()
        optimizer.step()
        print(loss)
        
    model.eval()
    gt_x = gt_x.astype(np.float32)
    torch_gt_x = torch.tensor(gt_x)
    with torch.no_grad():
        predict_y = model(torch_gt_x).numpy()
        
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y, label = 'data points')
    ax.plot(gt_x, gt_y, label = 'gt')
    ax.plot(gt_x, predict_y, label = 'predict')
    plt.savefig('deep.png')            
