# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 20:20:23 2019

@author: Shubham
"""

'''
0 count = 132915, 1 count = 11318, clearly unbalanced
'''

import pandas as pd
import numpy as np
from scipy.fftpack import fft, ifft
from glob import glob
import logging
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from torch.utils.data import Dataset, DataLoader
import warnings
from os.path import join as pjoin
from imblearn.combine import SMOTEENN
from sklearn.impute import SimpleImputer
from skimage import data, color

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

def combine_identity_transaction(mode='train'):
    print("Combining {} files".format(mode))
    merged_file_name = mode + '_merged.csv'
    merged_file_name = os.path.join('..', 'data', merged_file_name)
    print("File name is {}".format(merged_file_name))
    if len(glob(merged_file_name)) >= 1:
        print("Merged file already exists")
        return pd.read_csv(merged_file_name)
    csv_files = glob(mode + "_*.csv")
    csv_read_list = list(map(lambda x: pd.read_csv(x, index_col='TransactionID'),
                                    csv_files))
    combine_csv = csv_read_list[1].merge(csv_read_list[0], how='left', 
                               left_index=True, right_index=True)
    combine_csv.to_csv(merged_file_name, header=True)
    return pd.read_csv(merged_file_name)
    

def fft_viz(vec, name): 
    '''
    1-D fft and save the image for it to be used
    in Pytorch dataloadercc
    Input: pandas dataframe row
    Output: Grayscale numpy array
    '''
    fft_vec = fft(vec)
    fig = plt.figure(figsize=(1, 1))
    fig.canvas.draw()
    plt.plot(fft_vec)
#    fig.savefig(name + ".png")
    np_array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = color.rgb2gray(np_array.reshape(fig.canvas.get_width_height()[::-1] 
                        + (3,)))
    return data
    

def processDataFrame(train, test):
    y_train = train['isFraud'].copy()
    # Drop target, fill in NaNs 
    X_train = train.drop('isFraud', axis=1)
    X_test = test.copy()
    X_train = X_train.fillna(-999)
    X_test = X_test.fillna(-999)
    
    # label encoder
    for f in X_train.columns:
        if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(X_train[f].values) + list(X_test[f].values))
            X_train[f] = lbl.transform(list(X_train[f].values))
            X_test[f] = lbl.transform(list(X_test[f].values)) 
    return X_train, y_train, X_test
    

class TransactionDataset(Dataset):
    """
    Fraud Detection Pytorch dataset
    """
    def __init__(self, X_train, y_train, mode='N'):
        """
        Args:
        X_train, y_train: initial dataset
        mode: To apply SMOTE-ENN or not, 'N' is for normal, 'S' for smote
        """
        self.X_train = X_train
        self.y_train = y_train
        self.mode = mode
        if mode == 'S':
            smote_enn = SMOTEENN(random_state=0)
        else:
            X_train, y_train = X_train, y_train

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        name = "img_" + str(idx) +".png" 
        image = fft_viz(self.X_train.values[idx, :], name)
        image = np.expand_dims(image, 0).astype(float)
        isFraud = self.y_train[idx]
        sample = {'image': image, 'isFraud': isFraud}
        return sample
    
if __name__ == "__main__":
    modes = ['train', 'test']
    l = []
    for mode in modes:
        x = combine_identity_transaction(mode=mode)
        l.append(x)
    X_train, y_train, X_test = processDataFrame(l[0], l[1])
    fraud_dataset = TransactionDataset(X_train, y_train)
