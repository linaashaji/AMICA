"""
Created on Feb 7

@author: lachaji

"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class L2Loss(torch.nn.Module):

    def __init__(self):
        super(L2Loss, self).__init__()
    
        self.nb_ids = 10
        self.loss = nn.MSELoss()
    def forward(self, inputs, targets, mask):
        """
        Parameters
        ----------
        inputs : List(Tensor[batch_size * length, d])
        targets : List(Tensor[batch_size * length, d])
        
        
        To do: compare with weighted loss according to ID frequency 
        """
        total_loss = 0
        for inp, tar, m in zip(inputs, targets, mask):
            if(inp is not None):
                inp_m = inp[m]
                tar_m = tar[m]
                total_loss += self.loss(inp_m, tar_m) 
                
        return total_loss
 
   