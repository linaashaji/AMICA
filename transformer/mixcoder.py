"""
Created on Feb 7

@author: lachaji

"""

import torch.nn as nn
from .utils import clones

class Mixcoder(nn.Module):
    """
        Stack of N transfomers blocks + layerNom at the end
    """
    
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
    
    def forward(self, x, seed_length, src_mask=None, src_target_mask=None, mix_mask=None):
        for layer in self.layers:
            x = layer(x, seed_length, src_mask, src_target_mask, mix_mask)
        return self.norm(x)