# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf


def rescale(imgs):
    imgs = imgs/127.5
    imgs = imgs - 1.0
    return imgs

def get_data():
    (x, y), _ = tf.keras.datasets.mnist.load_data()
    shape = x.shape
    shape = [shape[0], shape[1], shape[2], 1]
    x = np.reshape(x, shape)
    x = rescale(x)
    x = x.transpose((0, 3, 1, 2))
    x = x.astype(np.float32)
    y = y.astype(np.long)    
    return x, y

if __name__ == '__main__':
    x, y = get_data()
    print(y.shape)
