"""
Created on Feb 7

@author: lachaji

"""

import torch
import torch.nn as nn

class Hidden2Normal(nn.Module):
    """
    Linear :  Hidden2normal
    """
    
    def __init__(self, d_model):
        super(Hidden2Normal, self).__init__()
        output_dim = 5
        self.proj = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        normal = self.proj(x)
        
        #print(normal.shape)
        
        # numerically stable output ranges
        normal[..., 2] = 0.01 + 0.2 * torch.sigmoid(normal[..., 2])  # sigma 1
        normal[..., 3] = 0.01 + 0.2 * torch.sigmoid(normal[..., 3])  # sigma 2
        normal[..., 4] = 0.7 * torch.sigmoid(normal[..., 4])  # rho

        return normal