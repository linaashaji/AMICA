"""
Created on Feb 7

@author: lachaji

"""

import torch
import torch.nn as nn

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
                if(inp.shape[0] > 0 and m.any()):
                    inp_m = inp[m]
                    tar_m = tar[m]
                    l = self.loss(inp_m, tar_m) 
                    total_loss += l
            else:
                print('The Input is None')
                
        return total_loss
 
   