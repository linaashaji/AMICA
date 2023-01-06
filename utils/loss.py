"""
Created on Feb 7

@author: lachaji

"""

import torch
import torch.nn as nn

class L2Loss(torch.nn.Module):

    def __init__(self, per_sequence_loss=False):
        super(L2Loss, self).__init__()
    
        self.nb_ids = 10
        self.loss = nn.MSELoss()
        self.per_sequence_loss = per_sequence_loss
        
    def forward(self, inputs, targets, mask, batch_size=None, batch_indices=None):
        """
        Parameters
        ----------
        inputs : List(Tensor[batch_size * length, d])
        targets : List(Tensor[batch_size * length, d])
        
        
        To do: compare with weighted loss according to ID frequency 
        """
        total_loss = 0
        if(self.per_sequence_loss):
            assert batch_size is not None
            assert batch_indices is not None
            psl = torch.zeros(batch_size).to(inputs[0].device)
        
            
            
        if(self.per_sequence_loss):
            for inp, tar, m, b_ind in zip(inputs, targets, mask, batch_indices):
                if(inp is not None):
                    if(inp.shape[0] > 0 and m.any()):
                        inp_m = inp[m]
                        tar_m = tar[m]
                        b_ind_m = b_ind[m]
                        l = self.loss(inp_m, tar_m)
                        total_loss += l.mean()
                        
                        for b in range(batch_size):
                            inp_m_ = inp_m[b_ind_m == b]
                            tar_m_ = tar_m[b_ind_m == b]
                            
                            if(inp_m_.shape[0] > 0):
                                sequence_loss = self.loss(inp_m_, tar_m_)
                                psl[b] += sequence_loss
                            
        else:
            for inp, tar, m in zip(inputs, targets, mask):
                if(inp is not None):
                    if(inp.shape[0] > 0 and m.any()):
                        inp_m = inp[m]
                        tar_m = tar[m]
                        l = self.loss(inp_m, tar_m)
                        total_loss += l
                    
        if(self.per_sequence_loss):
            return psl
        else:
            return total_loss

 
   