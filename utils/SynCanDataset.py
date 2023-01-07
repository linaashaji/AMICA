#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 18:36:49 2022

@author: lachaji
"""

import glob
import math
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


class SynCanDataset(Dataset):
  
    def __init__(self, ctx, split):

        data_path = ctx['data_path']
        window = ctx['window']
        stride = ctx['stride']
        
        path_list = sorted(glob.glob(f"{data_path}/{split}*.csv"))
        
        i = 0
        for path in path_list:
            df = pd.read_csv(path, names=['Label', 'Time', 'ID', 'Signal1', 'Signal2', 'Signal3', 'Signal4'])
            if (i == 0):
                dataframe = df.copy()
            else:
                dataframe = dataframe.append(df)
            i +=1
    
        dataframe.drop(0, inplace = True )
        
        
        self.data = self.sliding_window(dataframe, window, stride)
        
        if(split != 'test_real_life'):
            self.max_steps = self.calculate_maxpading(self.data)
        else: 
            self.max_steps = np.ones(10,) * window
            
        self.window = window
        self.signals = 4
        self.mask_value = 1.5
        self.mask_ratio = 0.15

            
    @staticmethod
    def calculate_maxpading(dlist):
        max_list = []
        for data_sample in dlist:
            l = np.array(data_sample["ID"].value_counts().sort_index())
            a = l[1]
            l[1:-1] = l[2:]
            l[-1] = a
            max_list.append(l)

        max_list = np.array(max_list)
        return np.max(max_list, axis = 0)
    
    @staticmethod
    def sliding_window(df, window, stride):
        datalist = []
        for i in range(0, len(df), stride):
            if(len(df.iloc[i:i+window]) == window):
                datalist.append(df.iloc[i:i+window].reset_index(drop=True))

        return datalist  
    
    @staticmethod
    def data_label(df):
        label = np.array(df['Label'], np.float64)
        
        if((label > 0).any()):
            label_ = 1
        else:
            label_ = 0
            
        return label_
    
    
    @staticmethod
    def time_padding(df_array, max_padding, stensor=False):
        pad_val = np.float64("nan")
        if(df_array.shape[0] == max_padding):
            return df_array
        else:
            diff = max_padding - len(df_array)
            
            if(stensor):
                to_concat = np.zeros(diff,)
            else:
                to_concat = np.zeros((diff, df_array.shape[-1]))
            to_concat.fill(pad_val)
            
            df_array = np.concatenate([ df_array, to_concat ], axis=0)
        
        return df_array


    @staticmethod
    def dimension_padding(df_array, max_padding):
        pad_val = np.float64("nan")
        if(df_array.shape[-1] == max_padding):
            return df_array
        else:
            diff = max_padding - df_array.shape[-1]
            
            to_concat = np.zeros((df_array.shape[0], diff))
            to_concat.fill(pad_val)
            
            df_array = np.concatenate([ df_array, to_concat ], axis=-1)
        
        return df_array

    
    @staticmethod
    def nan_to_num(x, replace_nan_with):
        x[x.isnan()] = replace_nan_with
        return x
    
    
    @staticmethod
    def bert_mask(x_orig, x_mask, mask_value, total_ratio):

        '''
        x_mask shape = (1, length)
        
        '''        
        
        x = x_orig.clone()
        non_indices = x_mask[0].nonzero()
        total_size = non_indices.size(0)

        length = x.shape[0]

        mask_ratio = 0.8
        random_ratio = 0.1
        
        perm = torch.randperm(total_size)[:int(total_size * total_ratio)]
        perm_length = len(perm)
        a = math.ceil(perm_length * mask_ratio)
        b = math.ceil(perm_length * random_ratio)
        
        perm_mask = perm[:a]
        perm_random = perm[a: a + b]
        
        perm_indices = torch.index_select(non_indices, 0, perm)
        perm_mask_indices = torch.index_select(non_indices, 0, perm_mask)
        perm_random_indices = torch.index_select(non_indices, 0, perm_random)
        

        x[perm_mask_indices] = mask_value
        for random_index in perm_random_indices:
            x[random_index] = torch.rand(1)
        
        mask_lm = torch.zeros(length, dtype=torch.bool)
        mask_lm[perm_indices] = True
        
        return x, mask_lm
    
    
    def __getitem__(self, index):
        return self.getitem(index)
    
    def __len__(self):
        return len(self.data)
    
    def getitem(self, index):

        df = self.data[index]
        label = self.data_label(df)
        d = {}
        d['label'] = label
        
        
        idrange = np.arange(1,11)
        for ecu_id in idrange:
            
            did = df[df.ID==f'id{str(ecu_id)}']
            values = np.array(did.dropna(axis=1).iloc[:,3:], dtype='float32')
            steps = np.array(did.index, dtype='float32')
            
            values = self.time_padding(values, self.max_steps[ecu_id - 1])
            steps = self.time_padding(steps, self.max_steps[ecu_id - 1], stensor=True)
            
            values_orig = self.dimension_padding(values, self.signals)
            
            values = torch.tensor(values, dtype = torch.float32)
            
            values_orig = torch.tensor(values_orig, dtype = torch.float32)
            steps = torch.tensor(steps, dtype = torch.float32)
            
            mask = values.isnan().any(dim=1)==0 # shape = (obs_length)
            mask = mask.unsqueeze(-2) # shape = (1, obs_length)
            
            values = self.nan_to_num(values, self.window*2)
            
            vmasked, mask_bert = self.bert_mask(values, mask, self.mask_value, self.mask_ratio)
            
            
            values_orig = self.nan_to_num(values_orig, self.window*2)
            steps = self.nan_to_num(steps, self.window*2)
            
            d[f'id{str(ecu_id)}'] = vmasked, steps, mask, mask_bert
            d[f'orig_id{str(ecu_id)}'] = values_orig
            


        return d