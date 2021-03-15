# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mnist import get_data
from network import Network

def eval_model(model, test_figs, test_labels):
    model.eval()
    batch_size = 100
    num_data = len(test_figs)
    num_iter = num_data // batch_size

    cnt = 0
    for i in range(num_iter):
        figs = torch.tensor(test_figs[i * batch_size: (i + 1) * batch_size])
        labels = test_labels[i * batch_size: (i + 1) * batch_size]
        with torch.no_grad():
            logits = model(figs)
            pred = np.argmax(logits.numpy(), axis = 1)
        cnt += np.sum(np.where(pred == labels, 1, 0))
    print('test_acc: {}'.format(float(cnt)/num_data))
            
if __name__ == '__main__':
    # -- parameter --
    train_num = 100
    batch_size = 100
    num_epoch = 100
    
    # -- data --
    x, y = get_data()
    train_figs, train_labels = x[:train_num], y[:train_num]
    test_figs, test_labels   = x[50000: ], y[50000: ]    

    # -- make model --
    model = Network()

    # -- criterion and optimizer --
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)    

    # -- train loop --
    num_one_epoch = train_num // batch_size
    idxs = [_ for _ in range(train_num)]
    
    for epoch in range(num_epoch):
        
        model.train()
        
        # -- shuffle data --
        tmp_idxs = np.random.permutation(idxs)
        train_figs   = train_figs[tmp_idxs]
        train_labels = train_labels[tmp_idxs]

        loss_epoch = 0
        for i in range(num_one_epoch):
            optimizer.zero_grad()
            inputs = torch.tensor(train_figs[i * batch_size: (i + 1) * batch_size])
            labels = torch.tensor(train_labels[i * batch_size: (i + 1) * batch_size])
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()/num_one_epoch
            
        print('epoch {}: loss = {}'.format(epoch, loss_epoch))
        eval_model(model, test_figs, test_labels)
