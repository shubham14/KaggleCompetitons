# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 14:47:38 2019

@author: Shubham
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from config import Config
from os.path import join as pjoin
import pickle as pkl
import datetime as dt
from sklearn import preprocessing

class DataHelper:
    '''
    DataHelper is for data viz and data processing
    cfg is the Config object with parameters
    '''
    def __init__(self, cfg):
        data_path = pjoin(cfg.BASE_DIR, "data")
        csv_names = ["building_metadata", "sample_submissions",
                     "test", "train", "weather_test", "weather_train"]
        csv_paths = glob(pjoin(data_path, "*.csv"))
        csv = list(map(lambda x: pd.read_csv(x), csv_paths))
        self.csv_dict = dict(zip(csv_names, csv))
        self.pklFileName = pjoin(cfg.BASE_DIR, 'csv_dicts.obj')
        
        # save in a pickle object which allows for fast loading 
        if len(glob(pjoin(cfg.BASE_DIR, "*.obj"))) == 0:
            print("Saving pickle file")
            pandas_pickle = open(self.pklFileName, 'w')
            pkl.dump(self.csv_dict, pandas_pickle)
        else:
            print("Loading from pickle file")
        
    def dataProc(self, df, metadata):
        '''
        df is train/test
        timestamp converted to ordinal numbers, can be used for regression
        one-hot encode categorical meter variable
        Joins on weather and metadata
        Transformation can take place on both train and test
        '''
        df['timestamp'] = pd.to_datetime(df['timestamp'], 
                                  format='%Y-%m-%d %H:%M:%S')
        df['timestamp'] = df['timestamp'].map(dt.datetime.toordinal)
        df['meter'] = pd.Categorical(df['meter'])
        dfDummies = pd.get_dummies(df['meter'], prefix='meter')
        df = pd.concat([df, dfDummies], axis=1) 
        df = df.drop(df['meter'])
        df_metadata = pd.merge(df, metadata, on='building_id', how='outer')
        return df, df_metadata
    
    def dataViz(self):
        self.csv_dict = pkl.load(self.pklFileName)
        df_metadata_train = self.csv_dict['building_metadata']
        df_train = self.csv_dict['train']
        df_test = self.csv_dict['test']
        df_train, df_metadata_train = self.dataProc(df_train, 
                                                    df_metadata_train)
        print('df_train columns are as follows : {}'.format(df_train.columns))
        print('df_test columns are as follows : {}'.format(df_test.columns))        
        
if __name__ == "__main__":
    cfg = Config()
    dataHelper = DataHelper(cfg)
        