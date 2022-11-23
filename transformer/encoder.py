"""
Created on Feb 7

@author: lachaji

"""

import torch.nn as nn
import torch.nn.functional as F

from .utils import clones

class Encoder(nn.Module):
    """
        Stack of N transfomers blocks + layerNom at the end
    """
    
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
    
    def forward(self, x, mask, look_ahead=None):
        for layer in self.layers:
            x = layer(x, mask, look_ahead)
        #return x
        return self.norm(x)