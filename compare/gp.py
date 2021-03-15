# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import theano.tensor as tt

from sample_data import gt, train_data

class Kernel(object):
    def __init__(self, xs, ell, factor, sigma):
        self.xs = xs
        self.ell = ell
        self.factor = factor * factor
        self.sigma = sigma
        self._get_matrix()
        
    def _basic_func(self, x, x_):
        norm = np.linalg.norm(x - x_)
        return self.factor * np.exp(-np.square(norm)/(2 * self.ell * self.ell))
    
    def _get_matrix(self):
        self.K_matrix = np.zeros((len(self.xs), len(self.xs)))
        for i, x in enumerate(self.xs):
            for j, x_ in enumerate(self.xs):
                self.K_matrix[i, j] = self._basic_func(x, x_)
        self.K_matrix = self.K_matrix + self.sigma * np.eye(len(self.xs))
        self.K_inv_matrix = np.linalg.inv(self.K_matrix)
        
    def get_k_vector(self, x):
        ret = np.zeros(len(self.xs))
        for i, x_ in enumerate(self.xs):
            ret[i] = self._basic_func(x, x_)
        return ret

    def get_k_sigma(self, x):
        return self.sigma + self._basic_func(x, x)


def get_variable_for_predict(kernel, x, ys):
    k_vec = kernel.get_k_vector(x)
    k_sigma = kernel.get_k_sigma(x)

    mu = np.dot(k_vec, np.dot(kernel.K_inv_matrix, ys))
    sigma = k_sigma - np.dot(k_vec, np.dot(kernel.K_inv_matrix, k_vec))

    return mu, sigma

if __name__ == '__main__':
    data_num = 30
    input_dim = 1
    
    gt_x, gt_y = gt()
    x, y = train_data(data_num, input_dim)


    model = pm.Model()
    with model:
        #ell    = pm.HalfNormal('ell', sd = 2.0)
        #factor = pm.HalfNormal('factor', sd = 2.0)
        ell    = pm.Uniform('ell', lower = 0.1, upper = 10.0)
        factor = pm.Uniform('factor', lower = 0.1, upper = 10.0)
        cov = factor ** 2 * pm.gp.cov.ExpQuad(input_dim, ell)
        
        sigma = pm.HalfNormal('sigma', sd = 1.0)
        total_sigma = sigma * np.eye(data_num) + cov(x.reshape(-1, input_dim))
        
        y_ = pm.MvNormal('y', mu = np.zeros(data_num), cov = total_sigma, observed = y)
        
    with model:
        #trace = pm.sample(100, tune = 3000, step = pm.NUTS())
        trace = pm.sample(10, tune = 250, step = pm.NUTS())

    ell_mean = np.mean(trace['ell'])
    factor_mean = np.mean(trace['factor'])
    sigma_mean = np.mean(trace['sigma'])

    K = Kernel(x, ell_mean, factor_mean, sigma_mean)

    predict_y = [get_variable_for_predict(K, _, y)[0] for _ in gt_x]
    predicts = [get_variable_for_predict(K, _, y) for _ in gt_x]
    lower = np.asarray([_[0] - 2 * _[1] for _ in predicts])
    upper = np.asarray([_[0] + 2 * _[1] for _ in predicts])
    
    lower = np.reshape(lower, (-1,))
    upper = np.reshape(upper, (-1,))    
    tmp_x = np.reshape(gt_x, (-1, ))
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y, label = 'data points')
    ax.plot(gt_x, gt_y, label = 'gt')
    ax.plot(gt_x, predict_y, label = 'predict')
    ax.fill_between(tmp_x, upper, lower, alpha = 0.5)    
    plt.savefig('gp.png')    
