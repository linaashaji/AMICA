"""
Created on Feb 7

@author: lachaji

"""

import torch.nn as nn

from. utils import clones

class Decoder(nn.Module):
    """
    N layer decoder with masking
    """
    
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
    
    def forward(self, memory, x, src_mask, tgt_mask, look_ahead=None):
        for layer in self.layers:
            x = layer(memory, x, src_mask, tgt_mask, look_ahead)
        return self.norm(x)
    
    
class TriDecoder(nn.Module):
    """
    N layer decoder with masking
    """
    
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
    
    def forward(self, memory_2, memory_1, x, memory_2_mask=None, memory_1_mask=None, x_mask=None):
        for layer in self.layers:
            x = layer(memory_2, memory_1, x, memory_2_mask, memory_1_mask, x_mask)
        return self.norm(x)