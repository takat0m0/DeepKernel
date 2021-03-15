# -*- coding:utf-8 -*-

import os
import sys
import pickle
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pyro
import pyro.contrib.gp as gp
import pyro.infer as infer

from mnist import get_data
from network import Network

def eval_model(gpmodule, test_figs, test_labels):

    batch_size = 100
    num_data = len(test_figs)
    num_iter = num_data // batch_size

    cnt = 0
    for i in range(num_iter):
        figs = torch.tensor(test_figs[i * batch_size: (i + 1) * batch_size])
        labels = torch.tensor(test_labels[i * batch_size: (i + 1) * batch_size])
        with torch.no_grad():
            f_loc, f_var = gpmodule(figs)
            pred = gpmodule.likelihood(f_loc, f_var)
        cnt += pred.eq(labels).long().sum().item()    
    print('test_acc: {}'.format(float(cnt)/num_data))

def plot_figs(gpmodule, test_figs):
    batch_size = 100
    figs = torch.tensor(test_figs[: batch_size])
    with torch.no_grad():
        f_loc, f_var = gpmodule(figs)
        pred = gpmodule.likelihood(f_loc, f_var)
        
    for i in range(batch_size):
        tmp = copy.deepcopy(test_figs[i])
        
        plt.subplot(1,2,1)
        plt.imshow(tmp.reshape(28, 28))
        plt.subplot(1,2,2)
        plt.bar(range(10), f_loc[:,i].detach(), yerr= f_var[:,i].detach())
        ax = plt.gca()
        ax.set_xticks(range(10))
        plt.xlabel("class")
        plt.savefig('image/figure'+ str(i) +'.png')
        plt.clf()
        
if __name__ == '__main__':
    # -- parameter --
    train_num = 10000
    batch_size = 100
    num_epoch = 100
    xu_num = 50
    
    # -- data --
    x, y = get_data()
    train_figs, train_labels = x[:train_num], y[:train_num]
    test_figs, test_labels   = x[50000: ], y[50000: ]    

    # -- make model --
    model = Network()

    # -- make gp
    rbf = gp.kernels.RBF(input_dim = 10, lengthscale = torch.ones(10))
    deep_kernel = gp.kernels.Warping(rbf, iwarping_fn = model)    
    idxs = [_ for _ in range(train_num)]
    tmp_idxs = np.random.permutation(idxs)
    train_figs, train_labels = train_figs[tmp_idxs], train_labels[tmp_idxs]
    xu = torch.tensor(train_figs[:xu_num]).clone()
    with open('xu.pickle', 'wb') as f:
        pickle.dump(xu, f)
        
    likelihood = gp.likelihoods.MultiClass(num_classes=10)
    latent_shape = torch.Size([10])
    gpmodule = gp.models.VariationalSparseGP(X = None, y=None,
                                             kernel = deep_kernel,
                                             Xu = xu,
                                             likelihood = likelihood,
                                             latent_shape = latent_shape,
                                             num_data = train_num,
                                             whiten = True,
                                             jitter = 2e-6)    
    # -- criterion and optimizer --
    optimizer = torch.optim.Adam(gpmodule.parameters(), lr=0.01)
    elbo = infer.TraceMeanField_ELBO()
    loss_fn = elbo.differentiable_loss

    # -- train loop --
    num_one_epoch = train_num // batch_size
    
    for epoch in range(num_epoch):
        
        # -- shuffle data --
        tmp_idxs = np.random.permutation(idxs)
        train_figs   = train_figs[tmp_idxs]
        train_labels = train_labels[tmp_idxs]

        loss_epoch = 0
        for i in range(num_one_epoch):

            inputs = torch.tensor(train_figs[i * batch_size: (i + 1) * batch_size])
            labels = torch.tensor(train_labels[i * batch_size: (i + 1) * batch_size])
            gpmodule.set_data(inputs, labels)
            optimizer.zero_grad()
            loss = loss_fn(gpmodule.model, gpmodule.guide)            
            loss.backward()
            optimizer.step()
            
            loss_epoch += loss.item()/num_one_epoch
            
        print('epoch {}: loss = {}'.format(epoch, loss_epoch))
        eval_model(gpmodule, test_figs, test_labels)
        plot_figs(gpmodule, test_figs)
        torch.save(gpmodule.state_dict(), './model.dump')
