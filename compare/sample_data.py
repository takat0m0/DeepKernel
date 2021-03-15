# -*- coding:utf-8 -*-

import os
import sys

import numpy as np

def _function(x):
    norm = np.linalg.norm(x, axis = 1)
    y = norm + np.sin(2 * norm)
    #y = np.square(x)
    return y

def gt(input_dim = 1):
    x = np.arange(-5, 5, 0.01)
    x = np.reshape(x, (-1, input_dim))
    return x, _function(x)

def train_data(data_num, input_dim = 1):
    np.random.seed(1234)
    x = np.random.uniform(-5, 5, (data_num, input_dim))
    y = _function(x)
    y_shape = y.shape
    y = y + np.random.normal(0, 0.2, y_shape)
    return x, y
