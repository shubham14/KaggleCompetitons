# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 10:20:28 2019

@author: Shubham
"""

# config file for external parameters 
class cfg:
    batch_size = 20
    EPOCHS = 10
    latent_dims = [20, 400]
    CUDA = False
    LOG_INTERVAL = 10
    img_channels = 3
    h_dim=1024
    z_dim=32
    image_channels = 1 # since it is a grayscale image
    device= 'cuda' # options are 'cuda' and 'cpu'